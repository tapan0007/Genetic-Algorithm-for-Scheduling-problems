// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh 
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <gmp.h>
#include <float.h>
#include "shader.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "dram.h"
#include "stat-tool.h"
#include "gpu-misc.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/cuda-sim.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "visualizer.h"
#include "../statwrapper.h"
#include "icnt_wrapper.h"
#include <string.h>
#include <limits.h>
#include "traffic_breakdown.h"
#include "shader_trace.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

bool gPrintQvalues = false;
    

/////////////////////////////////////////////////////////////////////////////

std::list<unsigned> shader_core_ctx::get_regs_written( const inst_t &fvt ) const
{
   std::list<unsigned> result;
   for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
      int reg_num = fvt.arch_reg.dst[op]; // this math needs to match that used in function_info::ptx_decode_inst
      if( reg_num >= 0 ) // valid register
         result.push_back(reg_num);
   }
   return result;
}


#define ALPHA 0.1

#define EXPLORATION_PERCENT 1.0

int gTotalGPUWideReward = 0;
double gTotalGPUWideDiscountedReward = 0.0;
int gReward = 1;
int gDiffReward = 0;
int gPenalty = 0;
int gDiffPenalty = 0;
int gSchedFreq = 1;
uint gMaxConsNoInstrs = 0;
uint gTwoPhase = 1;
uint gUniformBuckets = 0;
uint gBaseSched = 0;
uint gStnryFlag = 1;
bool gShareQvalueTableForAllSMs = false;

#define PROJECTION_LIMIT (1 << 30); // pow(10, 38)
int gMaxProjVal = 0xEFFFFFFF;
int gMinProjVal = 0xFFFFFFFF;
float gBeta = 0.1;
unsigned long long gBetaUpdatedAtCycle = 0;
float gGamma = 0.99;
float LAMBDA = 0.9;

valueMap* gQvalueTableForAllSMs[] = {0, 0, 0, 0};
valueUpdateMap* gQvalueUpdateTableForAllSMs[] = {0, 0, 0, 0};

#define USE_DUMMY_ACTION 0
#define USE_TB_ID_AS_ACTION 1
#define USE_WARP_ID_AS_ACTION 2
#define USE_CMD_PIPE_AS_ACTION 3
#define USE_TB_CMD_PIPE_AS_ACTION 4
#define USE_TB_WARP_ID_AS_ACTION 5
#define USE_TB_TYPE_AS_ACTION 6
#define USE_NUM_WARPS_AS_ACTION 7
#define USE_L1_BYPASS_AS_ACTION 8
#define USE_NUM_WARPS_AND_L1_BYPASS_AS_ACTION 9
#define USE_WHICH_SCHED_AS_ACTION 10
#define USE_WHICH_WARP_AS_ACTION 11
#define USE_NAM_ACTION 12
#define USE_WHICH_WARP_TYPE_AS_ACTION 13
#define USE_LRR_GTO_AS_ACTION 14

#define MAX_ACTIONS_OF_TYPE_TB_ID 9
#define MAX_ACTIONS_OF_TYPE_WARP_ID 25
#define MAX_ACTIONS_OF_TYPE_CMD_PIPE 5
#define MAX_ACTIONS_OF_TYPE_TB_ID_AND_CMD_PIPE 45
#define MAX_ACTIONS_OF_TYPE_TB_ID_AND_WARP_ID 225
#define MAX_ACTIONS_OF_TYPE_TB_TYPE 5
#define MAX_ACTIONS_OF_TYPE_NUM_WARPS 6
#define MAX_ACTIONS_OF_TYPE_BYPASS_L1 2
#define MAX_ACTIONS_OF_TYPE_NUM_WARPS_AND_BYPASS_L1 12
#define MAX_ACTIONS_OF_TYPE_WHICH_SCHED 5
#define MAX_ACTIONS_OF_TYPE_WHICH_WARP 9
#define MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE 5
#define MAX_ACTIONS_OF_LRR_GTO_TYPE 2
#define MAX_ACTIONS_OF_TYPE_NAM 3

#define SCHED_GTO_WARP 0
#define SCHED_YOUNGEST_WARP 1
#define SCHED_NEXT_WARP 2
#define SCHED_YOUNGEST_BARRIER_WARP 3
#define SCHED_YOUNGEST_FINISH_WARP 4
#define SCHED_OLDEST_SPLIT_WARP 5
#define SCHED_YOUNGEST_SPLIT_WARP 6
#define SCHED_OLDEST_LONG_LAT_MEM_WARP 7
#define SCHED_YOUNGEST_LONG_LAT_MEM_WARP 8

#define SCHED_GTO_WARP 0
#define SCHED_LRR_WARP 1
#define SCHED_YFB_WARP 2
#define SCHED_MFS_WARP 3
#define SCHED_FMS_WARP 4

#define SCHED_NO_INSTR 										0
#define SCHED_SP_INSTR 										1
#define SCHED_SFU_INSTR 									2
#define SCHED_GMEM_INSTR 									3  //global mem
#define SCHED_STC_MEM_INSTR 								4 //shared/texture/const mem
// #define SCHED_FUTURE_MEM_INSTR 								5 //shared/texture/const mem

#define SCHED_ALU_INSTR 1
#define SCHED_MEM_INSTR 2

#define SCHED_NO_TB 0
#define SCHED_FINISH_TB 1
#define SCHED_BARRIER_TB 2
#define SCHED_FASTEST_TB 3
#define SCHED_SLOWEST_TB 4

#define SCHED_ONE_WARP 		0
#define SCHED_TWO_WARPS 	1
#define SCHED_FOUR_WARPS 	2
#define SCHED_EIGHT_WARPS 	3
#define SCHED_SIXTEEN_WARPS 4
#define SCHED_ALL_WARPS 	5

#define DO_NOT_BYPASS_L1 0
#define BYPASS_L1 1

std::set<unsigned int> gBarrierTbIdSet;
std::set<unsigned int> gFinishTbIdSet;

unsigned int rl_scheduler::dRLActionTypes[] = {0xdeaddead, 0xdeaddead, 0xdeaddead, 0xdeaddead};

unsigned int rl_scheduler::gRandomSeed = 0;

rlEngine* gPrimaryRLEngine = 0;
rlEngine* gSecondaryRLEngine = 0;
valueMap* rl_scheduler::gPrimaryQvalues = 0;
unsigned long long rl_scheduler::gPrimaryQvalueArraySize = 0;
unsigned long long rl_scheduler::gPrimaryStateVal = 0;
unsigned int rl_scheduler::gPrimaryNumActions = 0;
valueMap* rl_scheduler::gSecondaryQvalues = 0;
unsigned long long rl_scheduler::gSecondaryQvalueArraySize = 0;
unsigned long long rl_scheduler::gSecondaryStateVal = 0;
unsigned int rl_scheduler::gSecondaryNumActions = 0;

Scoreboard* rl_scheduler::gScoreboard = 0;
simt_stack** rl_scheduler::gSimtStack = 0;
rl_scheduler* rl_scheduler::gCurrRLSchedulerUnit = 0;

#define NUM_WARPS_PER_SCHEDULER 24
#define DUMMY_WARP_ID 48 //assuming 48 warps from 0 to 47

#define MAX_NUM_TB_PER_SM 8
#define DUMMY_TB_ID MAX_NUM_TB_PER_SM //assuming 8 tbs from 0 to 7
#define MAX_NUM_WARP_PER_SM 48

bool gIPAWS = false;
unsigned int gSmToFinishFirstWarp = 0xdeaddead;
unsigned int gSlowestWarpId = 0xdeaddead;
bool gIPAWS_RecoverPhase = false;
bool gIPAWS_UseGTO = false;

bool rl_scheduler::gUsePrevQvalues = false;
bool rl_scheduler::gUseCMACFuncApprox = false;
bool rl_scheduler::gUseFeatureWeightFuncApprox = false;
bool gUseTbTypeAsAction = false;
bool gUseNumOfWarpsAsAction = false;
bool gUseBypassL1AsAction = false;
bool gUseNAMaction = false;
bool gUseCmdPipeTbTypeNumWarpsBypassAsAction = false;
bool gUseWhichSchedAsAction = false;
bool gUseWhichWarpAsAction = false;
bool gUseWhichWarpTypeAsAction = false;
bool gUseLrrGtoAsAction = false;
bool gGTOWarpAsAction = false;
bool gLRRWarpAsAction = false;
bool gYFBWarpAsAction = false;
bool gMFSWarpAsAction = false;
bool gFMSWarpAsAction = false;
//bool gYoungestWarpAsAction = false;
bool gUseMinAction = false;
bool gActorCriticMethod = false;
bool gNewActorCriticMethod = false;

unsigned int gGtoSchedCnt = 0;
unsigned int gLrrSchedCnt = 0;

unsigned int rl_scheduler::gNumWarpsExecutingMemInstrGPU = 0;
unsigned int rl_scheduler::gNumReqsInMemSchedQs = 0;
unsigned int rl_scheduler::gNumMemSchedQsLoaded = 0;
unsigned int rl_scheduler::gNumSpInstrIssued = 0;
unsigned int rl_scheduler::gNumSfuInstrIssued = 0;
unsigned int rl_scheduler::gNumGTCMemInstrIssued = 0;
unsigned int rl_scheduler::gNumL1Misses = 0;
unsigned int rl_scheduler::gNumSpInstrIssued1 = 0;
unsigned int rl_scheduler::gNumSfuInstrIssued1 = 0;
unsigned int rl_scheduler::gNumGTCMemInstrIssued1 = 0;
unsigned int rl_scheduler::gNumGTCMemInstrFinished = 0;
unsigned long long rl_scheduler::gNumGTCMemLatencyCycles = 0;

#define GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE 4
unsigned int* rl_scheduler::gGTCLongLatMemInstrCache = 0;
unsigned int rl_scheduler::gGTCLongLatMemInstrCacheIndex = 0;
unsigned long long rl_scheduler::gGTCTotalLatCycles = 0;
//bool rl_scheduler::gGTCLongLatMemInstrReady = false;

#define SFU_LONG_LAT_INSTR_CACHE_SIZE 4
unsigned int* rl_scheduler::gSFULongLatInstrCache = 0;
unsigned int rl_scheduler::gSFULongLatInstrCacheIndex = 0;
unsigned long long rl_scheduler::gSFUTotalLatCycles = 0;
bool rl_scheduler::gSFULongLatInstrReady = false;

#define DEFAULT_GLOBAL_MEM_LATENCY 300
#define LONG_LATENCY_MEM 200
#define LONG_LATENCY 50
#define NUM_SCHED_PER_SM 2

unsigned int* rl_scheduler::gNumWarpsExecutingMemInstr = 0;
unsigned int* rl_scheduler::gNumReadyMemInstrsWithSameTB = 0;
unsigned int* rl_scheduler::gLastMemInstrTB = 0;
unsigned int* rl_scheduler::gNumReadyMemInstrsWithSamePC = 0;
unsigned int* rl_scheduler::gNumInstrsIssued = 0;
unsigned int* rl_scheduler::gLastMemInstrPC = 0;
unsigned int* rl_scheduler::gNumWarpsWaitingAtBarrier = 0;
unsigned int* rl_scheduler::gNumWarpsFinished = 0;

unsigned int* gTBMinWarpTimeArray = 0;
unsigned int* gTBMaxWarpTimeArray = 0;
unsigned int* gTBProgressArray = 0;
unsigned int* gWarpProgressArray = 0;
unsigned int* gWarpTimeArray = 0;
unsigned int* gWarpDrainTimeArray = 0;
unsigned int gMinDrainTime = 0xFFFFFFFF;
unsigned int gMaxDrainTime = 0;
unsigned int gTotalNumWarpsFinished;
unsigned int gTotalDrainTime;
unsigned int* gSelectedTB = 0;
unsigned int* gSelectedWarp = 0;

unsigned int* rl_scheduler::gTBNumSpInstrsArray = 0;
unsigned int* rl_scheduler::gTBNumSfuInstrsArray = 0;
unsigned int* rl_scheduler::gTBNumMemInstrsArray = 0;
unsigned int rl_scheduler::gNumInstrsExecedByFinishedTBs = 0;
unsigned int rl_scheduler::gNumFinishedTBs = 0;

unsigned int gMaxNumInstrsExecedByTB = 0;
unsigned int gMinNumInstrsExecedByTB = 0xFFFFFFFF;

unsigned int gNumSMs = 0;
unsigned int gNumWarpsPerBlock = 0;
unsigned int gNumTBsPerSM = 0;
unsigned int gMaxNumResidentWarpsPerSm = 0;
unsigned int gMaxNumResidentWarpsPerSched = 0;
unsigned int gTotalNumOfTBsInGrid = 0;

unsigned int gExploitationCnt = 0;
unsigned int rl_scheduler::gExplorationCnt = 0;
unsigned int rl_scheduler::gSelectedActionVal = 12345678;

bool gLRRSched = false;
bool gGTOSched = false;
bool gRTOSched = false;
unsigned int gRTOSchedRandomPercent = 0;
unsigned int gRTOSchedRandomOrderCnt = 0;
unsigned int gRTOSchedGTOOrderCnt = 0;
bool gRLSched = false;
bool gRandomSched = false;

//bool gTwoRLEngines = false;
unsigned int gNumRLEngines = 1;

unsigned int gMemPipeLineStall = 0;
unsigned int gSfuPipeLineStall = 0;
unsigned int gSpPipeLineStall = 0;
unsigned int gMemSfuSpPipeLineStall = 0;
unsigned int gMemSpPipeLineStall = 0;
unsigned int gMemSfuPipeLineStall = 0;
unsigned int gSfuSpPipeLineStall = 0;

FILE* qvaluesFile = 0;
FILE* qvaluesFile2 = 0;
FILE* qvalueUpdateCntsFile = 0;
FILE* qvalueUpdateCntsFile2 = 0;

std::vector<std::map<unsigned int, unsigned int> > gSplitWarpDynamicIdMapVec;
std::vector<std::map<unsigned int, unsigned long long> > gNumCyclesStalledAtBarrierMapVec;
std::vector<std::map<unsigned int, unsigned int> > gNumWarpsAtBarrierMapVec;
std::vector<std::map<unsigned int, unsigned int> > gNumWarpsAtFinishMapVec;
std::set<unsigned int>gWarpsWaitingAtBarrierSet; //used by iPAWS
unsigned int* gWarpBarrierTimeArray = 0;
unsigned int* gWarpIssueStallArray = 0;
std::map<unsigned int, std::set<unsigned int> > gWarpsOfTbWaitingAtBarrierSetMap;

extern std::map<std::string, std::map<unsigned int, unsigned int> > gKernelBackEdgeSrcDstMap;
std::vector<unsigned int> gPhaseEndPCVec;
unsigned int gWrapAroundSrcPC = 0;
unsigned int gWrapAroundDestPC = 0;

unsigned int gPrintResultDirExt = 1;

static bool gPrintFlag = true;

unsigned int gSmId = 0xdeaddead;

unsigned int gMaxProg = 0xdeaddead;
unsigned int gMinProg = 0xdeaddead;
bool gPrintWarpProgress = false;

#define PROGRESS_DIFF_THRESHOLD (gNumWarpsPerBlock * 2)

#define LRR_SCHED 0
#define GTO_SCHED 1
#define OLDEST_SCHED 1
#define UNKNOWN_SCHED 2

std::vector<std::string> gModifiedAttrCombStrVec;
std::vector<unsigned long long> gModifiedAttrCombNumInstrVec;
std::vector<unsigned int> gModifiedAttrCombNumSMs;
std::vector<unsigned long long> gModifiedAttrCombMaxSimCyclesVec;
std::vector<std::map<unsigned int, std::vector<float> > > gWeightsVecMapVec;
std::vector<std::vector<std::map<unsigned long long, float> > > gQvalueSnapshotVecVec; //vector of vector of qvalue map samples
std::vector<std::vector<std::map<unsigned long long, unsigned int> > > gQvalueUpdateSnapshotVecVec; //vector of vector of qvalue updates map samples

unsigned int gNumSmGroups;
unsigned int gNumSmsPerGroup;

unsigned int gSnapshotSmId = 0xdeaddead;
unsigned int gSnapshotSchedId = 0xdeaddead;

std::map<unsigned int, std::string> gLastMemInstrMap;
extern unsigned int* gNumReqsInMemSchedArray;
bool gPrintDRAMInfo = false;

bool gBypassL1Cache = false;
bool gIsHighPrioMemReq = false;

std::set<unsigned int> gReadyTBIdSet;
unsigned int gCnt0 = 0;
unsigned int gCnt1 = 0;
unsigned int gCnt1_5 = 0;
unsigned int gCnt2 = 0;
unsigned int gCnt2_1 = 0;
unsigned int gCnt2_2 = 0;
unsigned int gCnt2_3 = 0;
unsigned int gCnt2_4 = 0;
unsigned int gCnt2_5 = 0;
unsigned int gCnt3 = 0;
unsigned int gCnt4 = 0;
unsigned int gCnt5 = 0;
unsigned int gCnt6 = 0;
unsigned int gCnt7 = 0;
unsigned int gCnt8 = 0;

bool gDebugQvalueUpdates = false;

std::map<unsigned int, unsigned int> gPossiblePrimaryActionCntMap;
std::vector<shd_warp_t*> gGTOWarpOrder;
unsigned int gNotSameWarpAsGTOCnt = 0;
unsigned int gSameWarpAsGTOCnt = 0;

unsigned long long gPrimaryActionCntSnapshotCycle = 0;
std::map<unsigned int, unsigned int> gPrimaryActionCntMap;
std::map<unsigned int, unsigned int> gSecondaryActionCntMap;
std::vector<std::map<unsigned int, unsigned int> > gPrimaryActionCntMapVec;

bool gPrintQvaluesFlag = true;

std::map<std::string, std::map<unsigned int, unsigned int> > gKernelStallCycleMapMap;
std::map<std::string, std::map<unsigned int, unsigned int> > gKernelRunCycleMapMap;

std::map<unsigned int, unsigned int> gBmStallCycleMap;
std::map<unsigned int, unsigned int> gBmRunCycleMap;

unsigned int numKernelsFinished = 0;

bool gPrintAttrValueCnts = false;
std::map<std::string, std::map<unsigned int, unsigned int> > gAttrNameValueCntMap;

std::map<unsigned int, std::vector<unsigned int> > gStateActionUpdateCntMap;
bool gStateActionUpdateCntMapEnabled = false;

std::map<unsigned int, unsigned long long> gInstrLatMap;
std::map<unsigned int, unsigned long long> gInstrNumExecMap;

std::set<std::string> gAttrNamesSet;

void gClearGlobals()
{
	gSplitWarpDynamicIdMapVec.clear();
    gSplitWarpDynamicIdMapVec.resize(gNumSMs);
	for (unsigned int id = 0; id < gNumSMs; id++)
	{
    	std::map<unsigned int, unsigned int> splitWarpDynamicIdMap;
		gSplitWarpDynamicIdMapVec[id] = splitWarpDynamicIdMap;
	}

	gNumCyclesStalledAtBarrierMapVec.clear();
    gNumCyclesStalledAtBarrierMapVec.resize(gNumSMs);

	gNumWarpsAtBarrierMapVec.clear();
    gNumWarpsAtBarrierMapVec.resize(gNumSMs);

	gNumWarpsAtFinishMapVec.clear();
    gNumWarpsAtFinishMapVec.resize(gNumSMs);

	gPhaseEndPCVec.clear();
	gModifiedAttrCombStrVec.clear();
	gModifiedAttrCombNumInstrVec.clear();
	gModifiedAttrCombNumSMs.clear();
	gModifiedAttrCombMaxSimCyclesVec.clear();
	gWeightsVecMapVec.clear();
	gWeightsVecMapVec.resize(gNumRLEngines);
	gQvalueSnapshotVecVec.clear();
	gQvalueSnapshotVecVec.resize(gNumRLEngines);
	gQvalueUpdateSnapshotVecVec.clear();
	gQvalueUpdateSnapshotVecVec.resize(gNumRLEngines);
	gLastMemInstrMap.clear();
	gPossiblePrimaryActionCntMap.clear();
	for (unsigned int i = 0; i < 50; i++)
		gPossiblePrimaryActionCntMap[i] = 0;
	gGTOWarpOrder.clear();
	gPrimaryActionCntMap.clear();
	gSecondaryActionCntMap.clear();
	gPrimaryActionCntMapVec.clear();
	gBmStallCycleMap.clear();
	gBmRunCycleMap.clear();
	gAttrNameValueCntMap.clear();
	gStateActionUpdateCntMap.clear();
	gInstrLatMap.clear();
	gInstrNumExecMap.clear();
}


#define NO_STALL 0
#define IDLE_STALL 1
#define SB_STALL 2
#define PIPE_STALL 3
void allocateWarpProgressArray(unsigned int numSMs)
{
    if (gWarpProgressArray == 0)
    {
        gWarpProgressArray = new unsigned int[numSMs * MAX_NUM_WARP_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_WARP_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_WARP_PER_SM + j;
                gWarpProgressArray[index] = 0;
            }
        }
    }
}

void allocateWarpBarrierTimeArray(unsigned int numSMs)
{
    if (gWarpBarrierTimeArray == 0)
    {
        gWarpBarrierTimeArray = new unsigned int[numSMs * MAX_NUM_WARP_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_WARP_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_WARP_PER_SM + j;
                gWarpBarrierTimeArray[index] = 0;
            }
        }
    }
}

void allocateWarpIssueStallArray(unsigned int numSMs)
{
    if (gWarpIssueStallArray == 0)
    {
        gWarpIssueStallArray = new unsigned int[numSMs * MAX_NUM_WARP_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_WARP_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_WARP_PER_SM + j;
                gWarpIssueStallArray[index] = 0;
            }
        }
    }
}

void allocateWarpTimeArray(unsigned int numSMs)
{
    if (gWarpTimeArray == 0)
    {
        gWarpTimeArray = new unsigned int[numSMs * MAX_NUM_WARP_PER_SM];
        gWarpDrainTimeArray = new unsigned int[numSMs * MAX_NUM_WARP_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_WARP_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_WARP_PER_SM + j;
                gWarpTimeArray[index] = 0;
                gWarpDrainTimeArray[index] = 0;
            }
        }
    }
}

void allocateTBProgressArray(unsigned int numSMs)
{
    if (gTBProgressArray == 0)
    {
        gTBProgressArray = new unsigned int[numSMs * MAX_NUM_TB_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_TB_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_TB_PER_SM + j;
                gTBProgressArray[index] = 0;
            }
        }
    }
}

void allocateTBMinAndMaxWarpTimeArray(unsigned int numSMs)
{
    if (gTBMinWarpTimeArray == 0)
    {
        gTBMinWarpTimeArray = new unsigned int[numSMs * MAX_NUM_TB_PER_SM];
        gTBMaxWarpTimeArray = new unsigned int[numSMs * MAX_NUM_TB_PER_SM];
        for (unsigned int i = 0; i < numSMs; i++)
        {
            for (unsigned int j = 0; j < MAX_NUM_TB_PER_SM; j++)
            {
                unsigned int index = i * MAX_NUM_TB_PER_SM + j;
                gTBMinWarpTimeArray[index] = 0xFFFFFFFF;
                gTBMaxWarpTimeArray[index] = 0;
            }
        }
    }
}

void initNumWarpsAtFinishMapVec(unsigned int numSMs)
{
    if (gNumWarpsAtFinishMapVec.size() != numSMs)
        gNumWarpsAtFinishMapVec.resize(numSMs);

	for (unsigned int id = 0; id < numSMs; id++)
	{
    	std::map<unsigned int, unsigned int>& numWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(id);
    	for (unsigned int i = 0; i < MAX_NUM_TB_PER_SM; i++)
        	numWarpsAtFinishMap[i] = 0;
	}
}

void initNumWarpsAtBarrierMapVec(unsigned int numSMs)
{
    if (gNumWarpsAtBarrierMapVec.size() != numSMs)
        gNumWarpsAtBarrierMapVec.resize(numSMs);

	for (unsigned int id = 0; id < numSMs; id++)
	{
   		std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(id);
   		for (unsigned int i = 0; i < MAX_NUM_TB_PER_SM; i++)
       		numWarpsAtBarrierMap[i] = 0;
	}
}


void initNumCyclesStalledAtBarrierMapVec(unsigned int numSMs)
{
    if (gNumCyclesStalledAtBarrierMapVec.size() != numSMs)
        gNumCyclesStalledAtBarrierMapVec.resize(numSMs);

	for (unsigned int id = 0; id < numSMs; id++)
	{
    	std::map<unsigned int, unsigned long long>& numCyclesStalledAtBarrierMap = gNumCyclesStalledAtBarrierMapVec.at(id);
    	for (unsigned int i = 0; i < MAX_NUM_TB_PER_SM; i++)
       		numCyclesStalledAtBarrierMap[i] = 0;
	}
}

void initSplitWarpDynamicIdMapVec(unsigned int numSMs)
{
    if (gSplitWarpDynamicIdMapVec.size() != numSMs)
        gSplitWarpDynamicIdMapVec.resize(numSMs);

	for (unsigned int id = 0; id < numSMs; id++)
	{
    	std::map<unsigned int, unsigned int> splitWarpDynamicIdMap;
		gSplitWarpDynamicIdMapVec[id] = splitWarpDynamicIdMap;
	}
}

void allocateNumWarpsExecutingMemInstrArr()
{
	if (rl_scheduler::gNumWarpsExecutingMemInstr == 0)
	{
		rl_scheduler::gNumWarpsExecutingMemInstr = new unsigned int[gNumSMs];
		for (unsigned int i = 0; i < gNumSMs; i++)
		    rl_scheduler::gNumWarpsExecutingMemInstr[i] = 0;
	}
}

void allocateNumReadyMemInstrsWithSameTBArr()
{
	if (rl_scheduler::gNumReadyMemInstrsWithSameTB == 0)
	{
		rl_scheduler::gNumReadyMemInstrsWithSameTB = new unsigned int[gNumSMs];
		rl_scheduler::gLastMemInstrTB = new unsigned int[gNumSMs];
		for (unsigned int i = 0; i < gNumSMs; i++)
		{
			rl_scheduler::gNumReadyMemInstrsWithSameTB[i] = 0;
			rl_scheduler::gLastMemInstrTB[i] = 0;
		}
	}
}

void allocateNumInstrsIssuedArr()
{
	if (rl_scheduler::gNumInstrsIssued == 0)
	{
		rl_scheduler::gNumInstrsIssued = new unsigned int[gNumSMs];
		for (unsigned int i = 0; i < gNumSMs; i++)
		{
		    rl_scheduler::gNumInstrsIssued[i] = 0;
		}
	}
}

void allocateNumReadyMemInstrsWithSamePCArr()
{
	if (rl_scheduler::gNumReadyMemInstrsWithSamePC == 0)
	{
		rl_scheduler::gNumReadyMemInstrsWithSamePC = new unsigned int[gNumSMs];
		rl_scheduler::gLastMemInstrPC = new unsigned int[gNumSMs];
		for (unsigned int i = 0; i < gNumSMs; i++)
		{
		    rl_scheduler::gNumReadyMemInstrsWithSamePC[i] = 0;
		    rl_scheduler::gLastMemInstrPC[i] = 0;
		}
	}
}

void allocateGTCLongLatMemInstrCacheArr()
{
	if (rl_scheduler::gGTCLongLatMemInstrCache == 0)
	{
		rl_scheduler::gGTCLongLatMemInstrCache = new unsigned int[gNumSMs * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE];
		for (unsigned int i = 0; i < gNumSMs * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE; i++)
		    rl_scheduler::gGTCLongLatMemInstrCache[i] = 0;
	}
}

void allocateSFULongLatInstrCacheArr()
{
	if (rl_scheduler::gSFULongLatInstrCache == 0)
	{
		rl_scheduler::gSFULongLatInstrCache = new unsigned int[gNumSMs * SFU_LONG_LAT_INSTR_CACHE_SIZE];
		for (unsigned int i = 0; i < gNumSMs * SFU_LONG_LAT_INSTR_CACHE_SIZE; i++)
		    rl_scheduler::gSFULongLatInstrCache[i] = 0;
	}
}
unsigned int checkAttributes(std::string xAttrStr, bool& warpIdAttr, bool& cmdPipeAttr, bool& tbIdAttr, bool& tbTypeAttr)
{
	unsigned int lNumRLEngines = 1;
	bool lTBWSeen = false;
	bool lNTFSeen = false;

	char lAttrsCharStr[1024];
	strcpy(lAttrsCharStr, xAttrStr.c_str());
	char* lToken = strtok(lAttrsCharStr, "_");
	while (lToken != NULL)
	{
	    if ((strcmp("ATBWB", lToken) == 0) || (strcmp("ATBWF", lToken) == 0) /*|| (strcmp("TBW", lToken) == 0)*/)
		{
			if (cmdPipeAttr == false)
			{
				rl_scheduler::dRLActionTypes[0] = USE_TB_TYPE_AS_ACTION;
				if (gUseTbTypeAsAction == false)
					rl_scheduler::dRLActionTypes[1] = USE_CMD_PIPE_AS_ACTION;
			}
	        tbTypeAttr = true;
		}
	    else if ((strstr(lToken, "SMNMIE") != 0)
	        || (strstr(lToken, "GNMIE") != 0)
	        || (strstr(lToken, "NRMI") != 0)
	        || (strstr(lToken, "NFMI") != 0)
	        || (strstr(lToken, "NFSFI") != 0)
	        || (strstr(lToken, "NRAI") != 0)
	        || (strstr(lToken, "NRSFI") != 0)
	        || (strstr(lToken, "NRSPI") != 0)
	        || (strstr(lToken, "NSW") != 0)
	        || (strstr(lToken, "NMQL") != 0)
	        || (strstr(lToken, "NMRQ") != 0)
	        || (strstr(lToken, "NAIPMI") != 0)
	        || (strstr(lToken, "NIPL1M") != 0)
	        || (strstr(lToken, "AGML") != 0)
	        || (strstr(lToken, "NRI") != 0)
	        || (strstr(lToken, "NWS") != 0)
	        || (strstr(lToken, "NWI") != 0)
	        || (strstr(lToken, "NPS") != 0)
	        || (strstr(lToken, "NMPS") != 0)
	        || (strstr(lToken, "NSFPS") != 0)
	        || (strstr(lToken, "NSPPS") != 0)
	        || (strstr(lToken, "NIW") != 0)
	        || (strstr(lToken, "STBRMI") != 0)
	        || (strstr(lToken, "SPCRMI") != 0)
	        || (strstr(lToken, "NRRMI") != 0)
	        || (strstr(lToken, "NRWMI") != 0)
	        || (strstr(lToken, "NRGMI") != 0)
	        || (strstr(lToken, "NRSTCMI") != 0)
	        || (strstr(lToken, "NRSMI") != 0)
	        || (strstr(lToken, "NRCMI") != 0)
	        || (strstr(lToken, "NRTMI") != 0)
	        || (strstr(lToken, "ICMP") != 0)
	        || (strstr(lToken, "L1MP") != 0)
	        || (strstr(lToken, "L2MP") != 0)
	        || (strcmp("RSPI", lToken) == 0)
	        || (strcmp("RSFI", lToken) == 0)
	        || (strcmp("RMI", lToken) == 0)
	        || (strcmp("RGMI", lToken) == 0)
	        || (strcmp("RLMI", lToken) == 0)
	        || (strcmp("RSMI", lToken) == 0)
	        || (strcmp("RCMI", lToken) == 0)
	        || (strcmp("RTMI", lToken) == 0)
	        || (strcmp("RSTCMI", lToken) == 0)
	        || (strcmp("RGCTRMI", lToken) == 0)
	        || (strcmp("RGCTMI", lToken) == 0)
	        || (strcmp("RGRMI", lToken) == 0)
	        || (strcmp("WP", lToken) == 0))
		{
			if (tbTypeAttr == false)
			{
				if (gUseTbTypeAsAction)
					rl_scheduler::dRLActionTypes[0] = USE_TB_TYPE_AS_ACTION;
				else
				{
					rl_scheduler::dRLActionTypes[0] = USE_CMD_PIPE_AS_ACTION;
					rl_scheduler::dRLActionTypes[1] = USE_TB_TYPE_AS_ACTION;
				}
			}
	        cmdPipeAttr = true;
		}
		else if ((strcmp("LTB", lToken) == 0) 
		      || (strcmp("FTB", lToken) == 0) 
			  || (strcmp("STB", lToken) == 0) 
			  || (strcmp("WTB", lToken) == 0) 
			  || (strcmp("TBWB", lToken) == 0) 
			  || (strcmp("TBWF", lToken) == 0)
			  || (strcmp("NWF", lToken) == 0)
			  || (strcmp("NWIB", lToken) == 0))
		{
			tbIdAttr = true;
			//rl_scheduler::gPrimaryActionType = USE_TB_ID_AS_ACTION;
			//rl_scheduler::gSecondaryActionType = USE_TB_ID_AS_ACTION;
			rl_scheduler::dRLActionTypes[0] = USE_TB_ID_AS_ACTION;
			rl_scheduler::dRLActionTypes[1] = USE_TB_ID_AS_ACTION;
		}
		else if (strcmp("LW", lToken) == 0)
		{
			warpIdAttr = true;
			//rl_scheduler::gPrimaryActionType = USE_WARP_ID_AS_ACTION;
			//rl_scheduler::gSecondaryActionType = USE_WARP_ID_AS_ACTION;
			rl_scheduler::dRLActionTypes[0] = USE_WARP_ID_AS_ACTION;
			rl_scheduler::dRLActionTypes[1] = USE_WARP_ID_AS_ACTION;
		}
		else if (strcmp("TBW", lToken) == 0)
			lTBWSeen = true;
		else if (strcmp("NTF", lToken) == 0)
			lNTFSeen = true;
		else if ((strstr(lToken, "ALPH") == 0) && (strstr(lToken, "XPL") == 0) && 
				 (strstr(lToken, "GAM") == 0) && (strstr(lToken, "RWRD") == 0) && 
				 (strstr(lToken, "PNLT") == 0) && (strstr(lToken, "QTBL") == 0) && 
				 (strstr(lToken, "SHDFR") == 0) && (strstr(lToken, "TWOPH") == 0) &&
				 (strstr(lToken, "QGTBL") == 0) && (strstr(lToken, "NOPS") == 0) &&
				 (strstr(lToken, "UBKTS") == 0) && (strstr(lToken, "BSHD") == 0) &&
				 (strstr(lToken, "STNRY") == 0) && (strstr(lToken, "BET") == 0) && (strstr(lToken, "PROJ") == 0))
		{
			assert(0);
			break;
		}

		lToken = strtok(NULL, "_");
	}

	if (warpIdAttr || tbIdAttr)
		return lNumRLEngines;

	if (gUseTbTypeAsAction)
	{
		rl_scheduler::dRLActionTypes[0] = USE_TB_TYPE_AS_ACTION;
		return lNumRLEngines;
	}
	else if (cmdPipeAttr && tbTypeAttr)
	{
		lNumRLEngines = 2;
		return lNumRLEngines;
	}

	if (lTBWSeen)
	{
		if (rl_scheduler::dRLActionTypes[0] == 0xdeaddead)
			rl_scheduler::dRLActionTypes[0] = USE_TB_TYPE_AS_ACTION;
		if (rl_scheduler::dRLActionTypes[1] == 0xdeaddead)
			rl_scheduler::dRLActionTypes[1] = USE_TB_TYPE_AS_ACTION;
	}
	else if (lNTFSeen)
	{
		if (gUseTbTypeAsAction)
			rl_scheduler::dRLActionTypes[0] = USE_TB_TYPE_AS_ACTION;
		else
		{
			if (rl_scheduler::dRLActionTypes[0] == 0xdeaddead)
				rl_scheduler::dRLActionTypes[0] = USE_CMD_PIPE_AS_ACTION;
			if (rl_scheduler::dRLActionTypes[1] == 0xdeaddead)
				rl_scheduler::dRLActionTypes[1] = USE_CMD_PIPE_AS_ACTION;
		}
	}
	return lNumRLEngines;
}

#define DEFAULT_NUM_BUCKETS THREE_BUCKETS

#define TWO_BUCKETS 2
#define THREE_BUCKETS 3
#define FOUR_BUCKETS 4
#define FIVE_BUCKETS 5

bool checkAttr(char* xToken, const char* xAttrName, unsigned int& xNumBuckets)
{
	xNumBuckets = DEFAULT_NUM_BUCKETS;
	bool lRetVal = false;
	if (strstr(xToken, xAttrName) != 0)
	{
		if (strlen(xToken) > strlen(xAttrName))
		{
			char* lNumBucketsStr = xToken + strlen(xAttrName);
			xNumBuckets = atoi(lNumBucketsStr);
		}
		lRetVal = true;
	}
	return lRetVal;
}

/*******************
void getModifiedAttr(const char* xBaseAttrName, const unsigned int xNumBuckets, char* xModifiedAttrName)
{
	unsigned int lNewNumBuckets = 2 + (random() % 3);

	sprintf(xModifiedAttrName, "%s%u", xBaseAttrName, lNewNumBuckets);
}

std::string gGenerateModifiedAttributeCombination(std::string xAttrStr)
{
	assert(0);

	char lAttrsCharStr[1024];
	strcpy(lAttrsCharStr, xAttrStr.c_str());
	char lModifiedAttrComb[1024];
	lModifiedAttrComb[0] = '\0';

	char* lToken = strtok(lAttrsCharStr, "_");
	while (lToken != NULL)
	{
		if (lModifiedAttrComb[0] != '\0')
			strcat(lModifiedAttrComb, "_");
		char lModifiedAttrName[1024];
		unsigned int lNumBuckets = DEFAULT_NUM_BUCKETS;
		if (checkAttr(lToken, "SMNMIE", lNumBuckets) == true) //if (strcmp("SMNMIE", lToken) == 0)
		{
			getModifiedAttr("SMNMIE", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "GNMIE", lNumBuckets) == true)
		{
			getModifiedAttr("GNMIE", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRAI", lNumBuckets) == true)
		{
			getModifiedAttr("NRAI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRSFI", lNumBuckets) == true)
		{
			getModifiedAttr("NRSFI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRSPI", lNumBuckets) == true)
		{
			getModifiedAttr("NRSPI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NSW", lNumBuckets) == true)
		{
			getModifiedAttr("NSW", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NMRQ", lNumBuckets) == true)
		{
			getModifiedAttr("NMRQ", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NMQL", lNumBuckets) == true)
		{
			getModifiedAttr("NMQL", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NAIPMI", lNumBuckets) == true)
		{
			getModifiedAttr("NAIPMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NAIPL1M", lNumBuckets) == true)
		{
			getModifiedAttr("NAIPL1M", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "AGML", lNumBuckets) == true)
		{
			getModifiedAttr("AGML", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NWI", lNumBuckets) == true)
		{
			getModifiedAttr("NWI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NPS", lNumBuckets) == true)
		{
			getModifiedAttr("NPS", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NMPS", lNumBuckets) == true)
		{
			getModifiedAttr("NMPS", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NSFPS", lNumBuckets) == true)
		{
			getModifiedAttr("NSFPS", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NSPPS", lNumBuckets) == true)
		{
			getModifiedAttr("NSPPS", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NIW", lNumBuckets) == true)
		{
			getModifiedAttr("NIW", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NWS", lNumBuckets) == true)
		{
			getModifiedAttr("NWS", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRI", lNumBuckets) == true)
		{
			getModifiedAttr("NRI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "STBRMI", lNumBuckets) == true)
		{
			getModifiedAttr("STBRMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "SPCRMI", lNumBuckets) == true)
		{
			getModifiedAttr("SPCRMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRRMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRRMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRWMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRWMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRGMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRGMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRSMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRSMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRSTCMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRSTCMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRCMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRCMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "NRTMI", lNumBuckets) == true)
		{
			getModifiedAttr("NRTMI", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "ICMP", lNumBuckets) == true)
		{
			getModifiedAttr("ICMP", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "L1MP", lNumBuckets) == true)
		{
			getModifiedAttr("L1MP", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (checkAttr(lToken, "L2MP", lNumBuckets) == true)
		{
			getModifiedAttr("L2MP", lNumBuckets, lModifiedAttrName);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (strstr(lToken, "ALPH") && (strlen(lToken) > 4))
		{
			unsigned int lAlpha = (5 + (random() % 5)) * 10;
			sprintf(lModifiedAttrName, "ALPH%u", lAlpha);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else if (strstr(lToken, "XPL") && (strlen(lToken) > 3))
		{
			unsigned int lExploration = (2 + (random() % 4)) * 4;
			if (lExploration == 8)
				lExploration = 12;
			else if (lExploration == 12)
				lExploration = 16;
			else if (lExploration == 16)
				lExploration = 20;
			else if (lExploration == 20)
				lExploration = 8;
			sprintf(lModifiedAttrName, "XPL%u", lExploration);
			strcat(lModifiedAttrComb, lModifiedAttrName);
		}
		else
			strcat(lModifiedAttrComb, lToken);

		lToken = strtok(NULL, "_");
	}
	std::string lModifiedAttrCombStr(lModifiedAttrComb);
	return lModifiedAttrCombStr;
}
****************/

double computeStdDev(std::vector<float>& valueVec, double& min, double& max, double& avg)
{
	double mean = 0.0;
	double sum = 0.0;
	for (unsigned int i = 0; i < valueVec.size(); i++)
	{
		double val = (double)valueVec[i];
		if (i == 0)
		{
			min = val;
			max = val;
		}
		else 
		{
			if (min > val)
				min = val;
			if (max < val)
				max = val;
		}
		sum += val;
	}
	mean = sum / valueVec.size();
	double sumDiffSq = 0.0;
	for (unsigned int i = 0; i < valueVec.size(); i++)
	{
		double diff = mean - (double)valueVec[i];
		double diffSq = diff * diff;
		sumDiffSq += diffSq;
	}
	double var = sumDiffSq / valueVec.size();
	double stdDev = sqrt(var);
	avg = mean;
	return stdDev;
}


void printWeights()
{
	printf("Func approx weight snapshots on SM %u\n", gSnapshotSmId);
	for (unsigned int lRlEngineNum = 0; lRlEngineNum < gWeightsVecMapVec.size(); lRlEngineNum++)
	{
		printf("BEGIN weights Engine %u\n", lRlEngineNum);
		std::map<unsigned int, std::vector<float> >& lWeightsVecMap = gWeightsVecMapVec[lRlEngineNum];

		for (std::map<unsigned int, std::vector<float> >::iterator iter = lWeightsVecMap.begin();
	     	iter != lWeightsVecMap.end();
		 	iter++)
		{
			unsigned int idx = iter->first;
			printf("%u", idx);
			std::vector<float>& wtVec = iter->second;
			for (std::vector<float>::iterator iter2 = wtVec.begin(); iter2 != wtVec.end(); iter2++)
			{
				printf(" %e", *(iter2));
			}
			printf("\n");
		}
		printf("END weights Engine %u\n", lRlEngineNum);

		bool printWeightsError = false;
		if (printWeightsError)
			printf("BEGIN weights error Engine %u\n", lRlEngineNum);

		std::vector<float> avgWtVec;
		std::vector<float> minWtVec;

		for (std::map<unsigned int, std::vector<float> >::iterator iter = lWeightsVecMap.begin();
	     	iter != lWeightsVecMap.end();
		 	iter++)
		{
			std::vector<float>& wtVec = iter->second;
			double sum = 0.0;
			float min = 0.0;
			for (unsigned int i = 0; i < wtVec.size(); i++)
			{
				float val = wtVec[i];
				if (i == 0)
					min = val;
				else if (min > val)
					min = val;
				sum += val;
			}
			minWtVec.push_back(min);
			float avg = sum / wtVec.size();
			avg = avg - min;
			avgWtVec.push_back(avg);
		}

		std::vector<float> totalErrorVec;
		for (std::map<unsigned int, std::vector<float> >::iterator iter = lWeightsVecMap.begin();
	     	iter != lWeightsVecMap.end();
		 	iter++)
		{
			unsigned int idx = iter->first;
			if (printWeightsError)
				printf("%u", idx);
			float avg = avgWtVec[idx];
			float min = minWtVec[idx];

			std::vector<float>& wtVec = iter->second;
			if (idx == 0)
			{
				for (unsigned int i = 0; i < wtVec.size(); i++)
					totalErrorVec.push_back(0.0);
			}
			for (unsigned int i = 0; i < wtVec.size(); i++)
			{
				float val = wtVec[i];
				val = val - min;
				double err;
				if (avg > val)
					err = avg - val;
				else
					err = val - avg;
				//err = err / avg;
				totalErrorVec[i] += err;
				if (printWeightsError)
					printf(" %e", err);
			}
			if (printWeightsError)
				printf("\n");
		}
		if (printWeightsError)
			printf("END weights error Engine %u\n", lRlEngineNum);

		printf("BEGIN weights error per snapshot Engine %u\n", lRlEngineNum);

		for (unsigned int i = 0; i < totalErrorVec.size(); i++)
		{
			printf("%u %e\n", i, totalErrorVec[i]);
		}
		printf("END weights error per snapshot Engine %u\n", lRlEngineNum);

/*
		printf("BEGIN weights std dev Engine %u\n", lRlEngineNum);
		for (std::map<unsigned int, std::vector<float> >::iterator iter = lWeightsVecMap.begin();
	     	iter != lWeightsVecMap.end();
		 	iter++)
		{
			unsigned int idx = iter->first;
			std::vector<float>& wtVec = iter->second;
			double min, max, avg;
			double stdDev = computeStdDev(wtVec, min, max, avg);
			double normMin = 1.0;
			double normMax = max/min;
			double normAvg = avg/min;
			printf("%u %f %f %f %f", idx, stdDev, normMin, normAvg, normMax);
			printf("\n");
		}
		printf("END weights std dev Engine %u\n", lRlEngineNum);
*/
	}
	gWeightsVecMapVec.clear();
	gWeightsVecMapVec.resize(gNumRLEngines);
}



void takeQvalueSnapshot(std::map<unsigned long long, float>& xValueMap, unsigned int xEngineNum)
{
	float totalQvalue = 0.0;
	assert(gQvalueSnapshotVecVec.size() > xEngineNum);
	//printf("Begin %u q value map snapshot\n", xEngineNum);
	for (std::map<unsigned long long, float>::iterator iter = xValueMap.begin(); iter != xValueMap.end(); iter++)
	{
		//printf("%llu %f\n", iter->first, iter->second);
		totalQvalue += iter->second;
	}
	printf("%llu: Engine %u Total q value = %f\n", gpu_sim_cycle, xEngineNum, totalQvalue);
	//printf("End %u q value map snapshot\n", xEngineNum);
	gQvalueSnapshotVecVec[xEngineNum].push_back(xValueMap);
}

void takeQvalueUpdateSnapshot(std::map<unsigned long long, unsigned int>& xValueUpdateMap, unsigned int xEngineNum)
{
	unsigned int totalUpdates = 0;
	//printf("Begin %u q value update map snapshot\n", xEngineNum);
	for (std::map<unsigned long long, unsigned int>::iterator iter = xValueUpdateMap.begin(); iter != xValueUpdateMap.end(); iter++)
	{
		//printf("%llu %u\n", iter->first, iter->second);
		totalUpdates += iter->second;
	}
	//printf("End %u q value update map snapshot\n", xEngineNum);
	printf("%llu: Engine %u Total Updates = %u\n", gpu_sim_cycle, xEngineNum, totalUpdates);

	assert(gQvalueUpdateSnapshotVecVec.size() > xEngineNum);
	gQvalueUpdateSnapshotVecVec[xEngineNum].push_back(xValueUpdateMap);
}

void printQvalueSamples()
{
	printf("Q value snapshots on SM %u\n", gSnapshotSmId);
	for (unsigned int lEngineNum = 0; lEngineNum < gQvalueSnapshotVecVec.size(); lEngineNum++)
	{
		std::map<unsigned long long, std::vector<float> > lQvaluesPerIndex; //map of index->vector of q value samples
		std::vector<std::map<unsigned long long, float> >& lQvalueSnapshotVec = gQvalueSnapshotVecVec[lEngineNum];
		unsigned int numSamples = lQvalueSnapshotVec.size();
		if (numSamples > 0)
		{
			std::map<unsigned long long, float>& lValueMap = lQvalueSnapshotVec[numSamples - 1];
			for (std::map<unsigned long long, float>::iterator iter = lValueMap.begin();
	   	  	iter != lValueMap.end();
			 	iter++)
			{
				std::vector<float> qValueVec;
				float defQValue = 1 / (1 - gGamma);
				for (unsigned int i = 0; i < numSamples; i++)
					qValueVec.push_back(defQValue);
				unsigned long long index = iter->first;
				lQvaluesPerIndex[index] = qValueVec;
			}
	
			for (unsigned int i = 0; i < numSamples; i++)
			{
				std::map<unsigned long long, float>& lValueMap = lQvalueSnapshotVec[i];
				for (std::map<unsigned long long, float>::iterator iter = lValueMap.begin();
			     	iter != lValueMap.end();
				 	iter++)
				{
					unsigned long long index = iter->first;
					float qValue = iter->second;
		
					std::vector<float>& qValueVec = lQvaluesPerIndex[index];
					qValueVec[i] = qValue;
				}
			}
		
			printf("BEGIN Engine %u q values per index:\n", lEngineNum);
			for (std::map<unsigned long long, std::vector<float> >::iterator iter = lQvaluesPerIndex.begin();
	   	  	iter != lQvaluesPerIndex.end();
			 	iter++)
			{
				unsigned long long index = iter->first;
				printf("%llu", index);
				std::vector<float>& qValueVec = iter->second;
				for (unsigned int i = 0; i < qValueVec.size(); i++)
				{
					printf(" %e", qValueVec[i]);
				}
				printf("\n");
			}
			printf("END Engine %u q values per index\n", lEngineNum);

			printf("BEGIN Engine %u q values mse per index:\n", lEngineNum);
			for (std::map<unsigned long long, std::vector<float> >::iterator iter = lQvaluesPerIndex.begin();
	   	  	iter != lQvaluesPerIndex.end();
			 	iter++)
			{
				unsigned long long index = iter->first;
				printf("%llu", index);
				std::vector<float>& qValueVec = iter->second;
				double sum = 0.0;
				for (unsigned int i = 0; i < qValueVec.size(); i++)
					sum += qValueVec[i];
				double avg = sum / qValueVec.size();
				for (unsigned int i = 0; i < qValueVec.size(); i++)
				{
					double err;
					if (avg > qValueVec[i])
						err = avg - qValueVec[i];
					else 
						err = qValueVec[i] - avg;
					//double mse = err * err;
					//printf(" %f", mse);
					printf(" %e", err);
				}
				printf("\n");
			}
			printf("END Engine %u q values mse per index\n", lEngineNum);

			printf("BEGIN Engine %u q values std dev:\n", lEngineNum);
			for (std::map<unsigned long long, std::vector<float> >::iterator iter = lQvaluesPerIndex.begin();
	   	  	iter != lQvaluesPerIndex.end();
			 	iter++)
			{
				unsigned long long index = iter->first;
				std::vector<float>& qValueVec = iter->second;
				double min, max, avg;
				double stdDev = computeStdDev(qValueVec, min, max, avg);
				double normMin = 1.0;
				double normMax = max/min;
				double normAvg = avg/min;
				printf("%llu %e %e %e %e", index, stdDev, normMin, normAvg, normMax);
			}
			printf("END Engine %u q values std dev\n", lEngineNum);
		}
		else
			printf("No Samples taken for q values\n");
	}
	gQvalueSnapshotVecVec.clear();
	gQvalueSnapshotVecVec.resize(gNumRLEngines);
}

void printQvalueUpdateSamples()
{
	printf("Q value update snapshots on SM %u\n", gSnapshotSmId);
	for (unsigned int lEngineNum = 0; lEngineNum < gQvalueUpdateSnapshotVecVec.size(); lEngineNum++)
	{
		std::map<unsigned long long, std::vector<unsigned int> > lQvalueUpdatesPerIndex; //vector of map of index->vector of q value update samples
		std::vector<std::map<unsigned long long, unsigned int> >& lQvalueUpdateSnapshotVec = gQvalueUpdateSnapshotVecVec[lEngineNum];

		unsigned int numSamples = lQvalueUpdateSnapshotVec.size();
		if (numSamples > 0)
		{
			std::map<unsigned long long, unsigned int>& lValueUpdateMap = lQvalueUpdateSnapshotVec[numSamples - 1];
	
			for (std::map<unsigned long long, unsigned int>::iterator iter = lValueUpdateMap.begin();
	     		iter != lValueUpdateMap.end();
		 		iter++)
			{
				std::vector<unsigned int> qValueUpdateVec;
				unsigned int defQValueUpdate = 0;
				for (unsigned int i = 0; i < numSamples; i++)
					qValueUpdateVec.push_back(defQValueUpdate);
				unsigned long long index = iter->first;
				lQvalueUpdatesPerIndex[index] = qValueUpdateVec;
			}
	
			for (unsigned int i = 0; i < numSamples; i++)
			{
				std::map<unsigned long long, unsigned int>& lValueUpdateMap = lQvalueUpdateSnapshotVec[i];
				for (std::map<unsigned long long, unsigned int>::iterator iter = lValueUpdateMap.begin();
		     		iter != lValueUpdateMap.end();
			 		iter++)
				{
					unsigned long long index = iter->first;
					unsigned int qValueUpdate = iter->second;
		
					std::vector<unsigned int>& qValueUpdateVec = lQvalueUpdatesPerIndex[index];
					qValueUpdateVec[i] = qValueUpdate;
				}
			}
		
			printf("BEGIN Engine %u q value updates per index:\n", lEngineNum);
			for (std::map<unsigned long long, std::vector<unsigned int> >::iterator iter = lQvalueUpdatesPerIndex.begin();
	     		iter != lQvalueUpdatesPerIndex.end();
		 		iter++)
			{
				unsigned long long index = iter->first;
				printf("%llu", index);
				std::vector<unsigned int>& qValueUpdateVec = iter->second;
				for (unsigned int i = 0; i < qValueUpdateVec.size(); i++)
				{
					printf(" %u", qValueUpdateVec[i]);
				}
				printf("\n");
			}
			printf("END Engine %u q value updates per index:\n", lEngineNum);
		}
		else
			printf("No Samples taken for q value updates\n");
	}
	gQvalueUpdateSnapshotVecVec.clear();
	gQvalueUpdateSnapshotVecVec.resize(gNumRLEngines);
}

shader_core_ctx::shader_core_ctx( class gpgpu_sim *gpu, 
                                  class simt_core_cluster *cluster,
                                  unsigned shader_id,
                                  unsigned tpc_id,
                                  const struct shader_core_config *config,
                                  const struct memory_config *mem_config,
                                  shader_core_stats *stats )
   : core_t( gpu, NULL, config->warp_size, config->n_thread_per_shader ),
     m_barriers( config->max_warps_per_shader, config->max_cta_per_core ),
     m_dynamic_warp_id(0)
{
    m_cluster = cluster;
    m_config = config;
    m_memory_config = mem_config;
    m_stats = stats;
    unsigned warp_size=config->warp_size;
    
    m_sid = shader_id;
    m_tpc = tpc_id;
    
    m_pipeline_reg.reserve(N_PIPELINE_STAGES);
    for (int j = 0; j<N_PIPELINE_STAGES; j++) {
        m_pipeline_reg.push_back(register_set(m_config->pipe_widths[j],pipeline_stage_name_decode[j]));
    }
    
    m_threadState = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);
    
    m_not_completed = 0;
    m_active_threads.reset();
    m_n_active_cta = 0;
    for ( unsigned i = 0; i<MAX_CTA_PER_SHADER; i++ ) 
        m_cta_status[i]=0;
    for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
        m_thread[i]= NULL;
        m_threadState[i].m_cta_id = -1;
        m_threadState[i].m_active = false;
    }
    
    // m_icnt = new shader_memory_interface(this,cluster);
    if ( m_config->gpgpu_perfect_mem ) {
        m_icnt = new perfect_memory_interface(this,cluster);
    } else {
        m_icnt = new shader_memory_interface(this,cluster);
    }
    m_mem_fetch_allocator = new shader_core_mem_fetch_allocator(shader_id,tpc_id,mem_config);
    
    // fetch
    m_last_warp_fetched = 0;
    
    #define STRSIZE 1024
    char name[STRSIZE];
    snprintf(name, STRSIZE, "L1I_%03d", m_sid);
    m_L1I = new read_only_cache( name,m_config->m_L1I_config,m_sid,get_shader_instruction_cache_id(),m_icnt,IN_L1I_MISS_QUEUE);
    
    m_warp.resize(m_config->max_warps_per_shader, shd_warp_t(this, warp_size));
    m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);
    
    //scedulers
    //must currently occur after all inputs have been initialized.
    std::string sched_config = m_config->gpgpu_scheduler_string;
    const concrete_scheduler scheduler = sched_config.find("lrr") != std::string::npos ?
                                         CONCRETE_SCHEDULER_LRR :
                                         sched_config.find("two_level_active") != std::string::npos ?
                                         CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE :
                                         sched_config.find("gto") != std::string::npos ?
                                         CONCRETE_SCHEDULER_GTO :
                                         sched_config.find("rto") != std::string::npos ?
                                         CONCRETE_SCHEDULER_RTO :
                                         sched_config.find("ipaws") != std::string::npos ?
                                         CONCRETE_SCHEDULER_IPAWS:
                                         sched_config.find("warp_limiting") != std::string::npos ?
                                         CONCRETE_SCHEDULER_WARP_LIMITING:
                                         sched_config.find("rl") != std::string::npos ?
                                         CONCRETE_SCHEDULER_REINFORCEMENT_LEARNING:
                                         sched_config.find("random") != std::string::npos ?
                                         CONCRETE_SCHEDULER_RANDOM:
                                         sched_config.find("random_ann") != std::string::npos?
                                         CONCRETE_SCHEDULER_RANDOM_ANN:
                                        NUM_CONCRETE_SCHEDULERS;
    assert ( scheduler != NUM_CONCRETE_SCHEDULERS );

    gNumSMs = this->get_config()->n_simt_clusters * this->get_config()->n_simt_cores_per_cluster;
	
	std::string lOrigAttrCombStr = m_config->rl_attrs;
	std::string lModifiedAttrCombStr;
	gNumSmGroups = 1;
	gNumSmsPerGroup = 15;

	if (m_sid == 0)
	{
		lModifiedAttrCombStr = lOrigAttrCombStr;
		gModifiedAttrCombStrVec.push_back(lModifiedAttrCombStr);
		gModifiedAttrCombNumInstrVec.push_back(0);
		gModifiedAttrCombNumSMs.push_back(0);
		gModifiedAttrCombMaxSimCyclesVec.push_back(0);
	}
	else
		lModifiedAttrCombStr = gModifiedAttrCombStrVec[m_sid % gNumSmGroups];

	unsigned int randomSeed = 0;

    allocateTBMinAndMaxWarpTimeArray(gNumSMs);
    allocateWarpTimeArray(gNumSMs);
    allocateWarpBarrierTimeArray(gNumSMs);
    allocateWarpIssueStallArray(gNumSMs);
    allocateWarpProgressArray(gNumSMs);
	allocateTBProgressArray(gNumSMs);
	initNumCyclesStalledAtBarrierMapVec(gNumSMs);
	initSplitWarpDynamicIdMapVec(gNumSMs);
    initNumWarpsAtBarrierMapVec(gNumSMs);
    initNumWarpsAtFinishMapVec(gNumSMs);
	allocateNumWarpsExecutingMemInstrArr();
	allocateNumReadyMemInstrsWithSameTBArr();
	allocateNumInstrsIssuedArr();
	allocateNumReadyMemInstrsWithSamePCArr();
	allocateGTCLongLatMemInstrCacheArr();
	allocateSFULongLatInstrCacheArr();

    for (int i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
        switch( scheduler )
        {
            case CONCRETE_SCHEDULER_LRR:
                gLRRSched = true;
                schedulers.push_back(
                    new lrr_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE:
                schedulers.push_back(
                    new two_level_active_scheduler( m_stats,
                                                    this,
                                                    m_scoreboard,
                                                    m_simt_stack,
                                                    &m_warp,
                                                    &m_pipeline_reg[ID_OC_SP],
                                                    &m_pipeline_reg[ID_OC_SFU],
                                                    &m_pipeline_reg[ID_OC_MEM],
                                                    i,
                                                    config->gpgpu_scheduler_string
                                                  )
                );
                break;
            case CONCRETE_SCHEDULER_GTO:
                gGTOSched = true;
                schedulers.push_back(
                    new gto_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_RTO:
			{
                gRTOSched = true;
                schedulers.push_back(
                    new rto_scheduler( m_stats,
                                       this,
                    	                   m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
				char str[20];
				strcpy(str, sched_config.c_str());
				char* ptr = strstr(str, "rto");
				assert (ptr == str);
				ptr = ptr + 3;
				gRTOSchedRandomPercent = atoi(ptr);
				printf("gRTOSchedRandomPercent = %u\n", gRTOSchedRandomPercent);
                break;
			}
            case CONCRETE_SCHEDULER_IPAWS:
			{
				if (gIPAWS == false)
				{
					printf("Running IPAWS\n");
                	gIPAWS = true;
				}
                schedulers.push_back(
                    new ipaws_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i
                                     )
                );
                break;
			}
            case CONCRETE_SCHEDULER_WARP_LIMITING:
                schedulers.push_back(
                    new swl_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i,
                                       config->gpgpu_scheduler_string
                                     )
                );
                break;
            case CONCRETE_SCHEDULER_REINFORCEMENT_LEARNING:
            {
				if (gRLSched == false)
				{
					if (sched_config.find("rl_pq") != std::string::npos)
						rl_scheduler::gUsePrevQvalues = true;
					else if (sched_config.find("rl_fa") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						printf("Using Feature Weight Func Approx\n");
					}
					else if (sched_config.find("rl_cmac") != std::string::npos)
					{
						rl_scheduler::gUseCMACFuncApprox = true;
						printf("Using CMAC Func Approx\n");
					}
					else if (sched_config.find("rl_ctsb") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						gUseCmdPipeTbTypeNumWarpsBypassAsAction = true;

						printf("cmd pipe as action\n");
						printf("tb type as action\n");
						printf("Num of warps as action\n");
						printf("bypass L1 as action\n");
					}
					else if (sched_config.find("rl_ws") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						if (sched_config.find("rl_ws1") != std::string::npos)
						{
							gGTOWarpAsAction = true;
							printf("Using gGTOWarpAsAction as action\n");
						}
						else if (sched_config.find("rl_ws2") != std::string::npos)
						{
							gLRRWarpAsAction = true;
							printf("Using gLRRWarpAsAction as action\n");
						}
						else if (sched_config.find("rl_ws3") != std::string::npos)
						{
							gYFBWarpAsAction = true;
							printf("Using gYFBWarpAsAction as action\n");
						}
						else if (sched_config.find("rl_ws4") != std::string::npos)
						{
							gMFSWarpAsAction = true;
							printf("Using gMFSWarpAsAction as action\n");
						}
						else if (sched_config.find("rl_ws5") != std::string::npos)
						{
							gFMSWarpAsAction = true;
							printf("Using gFMSWarpAsAction as action\n");
						}

						gUseWhichSchedAsAction = true;
						printf("which sched as action\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
					}
					else if (sched_config.find("rl_ww") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						gUseWhichWarpAsAction = true;
						printf("which warp as action\n");
					}
					else if (sched_config.find("rl_wt") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						gUseWhichWarpTypeAsAction = true;
						printf("which warp type as action\n");
					}
					else if (sched_config.find("rl_lg") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						gUseLrrGtoAsAction = true;
						printf("lrr gto as action\n");
					}
					else if (sched_config.find("rl_swl_bpl") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						gUseNumOfWarpsAsAction = true;
						gUseBypassL1AsAction = true;
						printf("Num of warps as action\n");
						printf("bypass L1 as action\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
					}
					else if (sched_config.find("rl_swl") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						gUseNumOfWarpsAsAction = true;
						printf("Num of warps as action\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
					}
					else if (sched_config.find("rl_bpl") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						gUseBypassL1AsAction = true;
						printf("bypass L1 as action\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
					}
					else if (sched_config.find("rl_tb") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;

						gUseTbTypeAsAction = true;
						printf("tb type as action\n");
					}
					else if (sched_config.find("rl_nam") != std::string::npos)
					{
						gUseNAMaction = true;
						printf("Using NAM action action\n");
					}
					else if (sched_config.find("rl_fnam") != std::string::npos)
					{
						gUseNAMaction = true;
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						printf("Using Feature Weight Func Approx\n");
						printf("Using NAM action action\n");
					}
					else if (sched_config.find("rl_min") != std::string::npos)
					{
						gUseMinAction = true;
						printf("Using min q value action\n");
					}
					else if (sched_config.find("rl_fm") != std::string::npos)
					{
						rl_scheduler::gUsePrevQvalues = true;
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						gUseMinAction = true;
						printf("Using Feature Weight Func Approx\n");
						printf("Using min q value action\n");
					}
					else if (sched_config.find("rl_nacf") != std::string::npos)
					{
						gNewActorCriticMethod = true;
						printf("Using NewActor Critic Method\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						printf("Using Feature Weight Func Approx\n");
					}
					else if (sched_config.find("rl_af") != std::string::npos)
					{
						gActorCriticMethod = true;
						printf("Using Actor Critic Method\n");
						rl_scheduler::gUseFeatureWeightFuncApprox = true;
						printf("Using Feature Weight Func Approx\n");
					}
					else if (sched_config.find("rl_ac") != std::string::npos)
					{
						gActorCriticMethod = true;
						printf("Using Actor Critic Method\n");
					}
					else 
						assert(0);

					if (rl_scheduler::gUsePrevQvalues == true)
						printf("Using prev q value table\n");
					else
						printf("NOT Using prev q value table\n");

    				bool warpIdAttr = false;
    				bool cmdPipeAttr = false;
    				bool tbIdAttr = false;
    				bool tbTypeAttr = false;
					if ((gUseNumOfWarpsAsAction == false) && 
						(gUseBypassL1AsAction == false) && 
						(gUseWhichSchedAsAction == false) &&
						(gUseNAMaction == false) &&
						(gUseLrrGtoAsAction == false) &&
						(gUseWhichWarpTypeAsAction == false) &&
						(gUseWhichWarpAsAction == false))
					{
						gNumRLEngines = checkAttributes(lModifiedAttrCombStr, warpIdAttr, cmdPipeAttr, tbIdAttr, tbTypeAttr);
						if (gUseCmdPipeTbTypeNumWarpsBypassAsAction)
						{
							gNumRLEngines = 4;
							rl_scheduler::dRLActionTypes[0] = USE_CMD_PIPE_AS_ACTION;
							rl_scheduler::dRLActionTypes[1] = USE_TB_TYPE_AS_ACTION;
							rl_scheduler::dRLActionTypes[2] = USE_NUM_WARPS_AS_ACTION;
							rl_scheduler::dRLActionTypes[3] = USE_L1_BYPASS_AS_ACTION;
						}
						else if (gUseTbTypeAsAction)
						{
						}
						else
						{
							if (gNumRLEngines == 2)
							{
								rl_scheduler::dRLActionTypes[0] = USE_CMD_PIPE_AS_ACTION;
								rl_scheduler::dRLActionTypes[1] = USE_TB_TYPE_AS_ACTION;
		
								if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
									printf("Primary action type = TB_TYPE\n");
								else if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
									printf("Primary action type = CMD_PIPE\n");
					
								if (rl_scheduler::dRLActionTypes[1] == USE_TB_TYPE_AS_ACTION)
									printf("Secondary action type = TB_TYPE\n");
								else if (rl_scheduler::dRLActionTypes[1] == USE_CMD_PIPE_AS_ACTION)
									printf("Secondary action type = CMD_PIPE\n");
							}
							else
							{
								if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
									printf("Primary action type = TB_TYPE\n");
								else if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
									printf("Primary action type = CMD_PIPE\n");
							}
						}
					}
					gQvalueSnapshotVecVec.resize(gNumRLEngines);
					gQvalueUpdateSnapshotVecVec.resize(gNumRLEngines);
					gWeightsVecMapVec.resize(gNumRLEngines);
					/*
					else if (gUseCmdPipeTbTypeNumWarpsBypassAsAction)
					{
						//create four RL engines
						gNumRLEngines = 4;
						rl_scheduler::dRLActionTypes[0] = USE_CMD_PIPE_AS_ACTION;
						rl_scheduler::dRLActionTypes[1] = USE_TB_TYPE_AS_ACTION;
						rl_scheduler::dRLActionTypes[2] = USE_NUM_WARPS_AS_ACTION;
						rl_scheduler::dRLActionTypes[3] = USE_L1_BYPASS_AS_ACTION;
					}
					*/
				}
					
                gRLSched = true;
                rl_scheduler* rlSched = 
                    new rl_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i,
									   gNumRLEngines);
                schedulers.push_back(rlSched);

                if ((m_sid == 0) && (i == 0))
				{
                    printf("Running RL scheduler\n");
				}

                if ((m_sid == 0) && (i == 0))
                {
                    unsigned int seed = 0;
                    srandom(seed);
                }

                rlSched->dAttrString = lModifiedAttrCombStr;

                break;
            }
            case CONCRETE_SCHEDULER_RANDOM:
            {
                random_scheduler* random_sched = new random_scheduler( m_stats,
                                       this,
                                       m_scoreboard,
                                       m_simt_stack,
                                       &m_warp,
                                       &m_pipeline_reg[ID_OC_SP],
                                       &m_pipeline_reg[ID_OC_SFU],
                                       &m_pipeline_reg[ID_OC_MEM],
                                       i);
                schedulers.push_back(random_sched);
				gRandomSched = true;
                if ((m_sid == 0) && (i == 0))
                {
                    unsigned int seed = 0xFFFFFFFF - (unsigned int) time(0);
                    printf("Using random seed %u\n", seed);
                    srandom(seed);
					randomSeed = seed;
                }
                break;
            }
            default:
                abort();
        };

		if (gRLSched)
		{
		}
		else
		{
        	if ((m_sid == 0) && (i == 0))
        	{
				if (gRandomSched)
            		printf("RESULT_DIR_EXT=\n");
				else
            		printf("RESULT_DIR_EXT=_%s\n", sched_config.c_str());
        	}
		}
    }
    
    for (unsigned i = 0; i < m_warp.size(); i++) {
        //distribute i's evenly though schedulers;
        schedulers[i%m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(i);
    }
    for ( int i = 0; i < m_config->gpgpu_num_sched_per_core; ++i ) {
        schedulers[i]->done_adding_supervised_warps();
    }
    
    //op collector configuration
    enum { SP_CUS, SFU_CUS, MEM_CUS, GEN_CUS };
    m_operand_collector.add_cu_set(SP_CUS, m_config->gpgpu_operand_collector_num_units_sp, m_config->gpgpu_operand_collector_num_out_ports_sp);
    m_operand_collector.add_cu_set(SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu, m_config->gpgpu_operand_collector_num_out_ports_sfu);
    m_operand_collector.add_cu_set(MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem, m_config->gpgpu_operand_collector_num_out_ports_mem);
    m_operand_collector.add_cu_set(GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen, m_config->gpgpu_operand_collector_num_out_ports_gen);
    
    opndcoll_rfu_t::port_vector_t in_ports;
    opndcoll_rfu_t::port_vector_t out_ports;
    opndcoll_rfu_t::uint_vector_t cu_sets;
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
        cu_sets.push_back((unsigned)SP_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
        cu_sets.push_back((unsigned)SFU_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
        cu_sets.push_back((unsigned)MEM_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);                       
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }   
    
    
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen; i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
        in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
        in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
        cu_sets.push_back((unsigned)GEN_CUS);   
        m_operand_collector.add_port(in_ports,out_ports,cu_sets);
        in_ports.clear(),out_ports.clear(),cu_sets.clear();
    }
    
    m_operand_collector.init( m_config->gpgpu_num_reg_banks, this );
    
    // execute
    m_num_function_units = m_config->gpgpu_num_sp_units + m_config->gpgpu_num_sfu_units + 1; // sp_unit, sfu, ldst_unit
    //m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
    //m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];
    
    //m_fu = new simd_function_unit*[m_num_function_units];
    
    for (int k = 0; k < m_config->gpgpu_num_sp_units; k++) {
        m_fu.push_back(new sp_unit( &m_pipeline_reg[EX_WB], m_config, this ));
        m_dispatch_port.push_back(ID_OC_SP);
        m_issue_port.push_back(OC_EX_SP);
    }
    
    for (int k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
        m_fu.push_back(new sfu( &m_pipeline_reg[EX_WB], m_config, this ));
        m_dispatch_port.push_back(ID_OC_SFU);
        m_issue_port.push_back(OC_EX_SFU);
    }
    
    m_ldst_unit = new ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id );
    m_fu.push_back(m_ldst_unit);
    m_dispatch_port.push_back(ID_OC_MEM);
    m_issue_port.push_back(OC_EX_MEM);
    
    assert(m_num_function_units == m_fu.size() and m_fu.size() == m_dispatch_port.size() and m_fu.size() == m_issue_port.size());
    
    //there are as many result buses as the width of the EX_WB stage
    num_result_bus = config->pipe_widths[EX_WB];
    for(unsigned i=0; i<num_result_bus; i++){
        this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
    }
    
    m_last_inst_gpu_sim_cycle = 0;
    m_last_inst_gpu_tot_sim_cycle = 0;

    whichSchedFirst = 0;
}

void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed ) 
{
   if( reset_not_completed ) {
       m_not_completed = 0;
       m_active_threads.reset();
   }
   for (unsigned i = start_thread; i<end_thread; i++) {
      m_threadState[i].n_insn = 0;
      m_threadState[i].m_cta_id = -1;
   }
   for (unsigned i = start_thread / m_config->warp_size; i < end_thread / m_config->warp_size; ++i) {
      m_warp[i].reset();
      m_simt_stack[i]->reset();
   }
}

extern char* program_invocation_short_name;
void createInstrPhaseMap(std::string kernelName)
{
    std::map<unsigned int, unsigned int> backEdgeSrcDstMap = gKernelBackEdgeSrcDstMap[kernelName];
    if (backEdgeSrcDstMap.size() == 0)
    {
        printf("%s does not have any backedges\n", kernelName.c_str());
        return;
    }
    for (std::map<unsigned int, unsigned int>::iterator iter = backEdgeSrcDstMap.begin();
         iter != backEdgeSrcDstMap.end();
         iter++)
    {
        unsigned int backEdgeSrcPC = iter->first;
        unsigned int backEdgeDstPC = iter->second;
        unsigned int beginPC;
        unsigned int endPC;

        if (backEdgeSrcPC < backEdgeDstPC)
        {
            beginPC = backEdgeSrcPC;
            endPC = backEdgeDstPC;
        }
        else
        {
            beginPC = backEdgeDstPC;
            endPC = backEdgeSrcPC;
        }

        bool addPhase = true;

        for (std::map<unsigned int, unsigned int>::iterator iter2 = backEdgeSrcDstMap.begin();
             iter2 != backEdgeSrcDstMap.end();
             iter2++)
        {
            unsigned int backEdgeSrcPC2 = iter2->first;
            unsigned int backEdgeDstPC2 = iter2->second;
            unsigned int beginPC2;
            unsigned int endPC2;
            if (backEdgeSrcPC2 < backEdgeDstPC2)
            {
                beginPC2 = backEdgeSrcPC2;
                endPC2 = backEdgeDstPC2;
            }
            else
            {
                beginPC2 = backEdgeDstPC2;
                endPC2 = backEdgeSrcPC2;
            }

            if ((beginPC != beginPC2) && (endPC != endPC2))
            {
                if ((beginPC > beginPC2) && (beginPC < endPC2))
                {
                    addPhase = false;
                    printf("%u to %u is a not phase, conflicts with %u to %u\n", beginPC, endPC, beginPC2, endPC2);
                }
            }
        }
        if (addPhase == true)
        {
            printf("%u to %u is a phase\n", beginPC, endPC);
            gPhaseEndPCVec.push_back(endPC);
			printf("prog exec name %s\n", program_invocation_short_name);
			printf("prog exec name %s\n", program_invocation_name);
			printf("prog exec name %s/%s\n", get_current_dir_name(), program_invocation_name);
			if ((strstr(program_invocation_short_name, "warpDiv_static") != 0) || 
				(strstr(get_current_dir_name(), "warpDiv_static") != 0))
			{
				if (endPC > gWrapAroundSrcPC)
					gWrapAroundSrcPC = endPC;
				if (beginPC > gWrapAroundDestPC)
					gWrapAroundDestPC = beginPC;
			}
        }
    }
	if (gWrapAroundDestPC != 0)
		printf("wrap around dest pc = %u\n", gWrapAroundDestPC);
	if (gWrapAroundSrcPC != 0)
		printf("wrap around src pc = %u\n", gWrapAroundSrcPC);
    if (gPhaseEndPCVec.size() > 0)
        std::sort(gPhaseEndPCVec.begin(), gPhaseEndPCVec.end());
}

unsigned int gGetNumWarpsOfInterest(unsigned int smId)
{
	unsigned int maxIssueStallWarpId = 0xdeaddead;
	unsigned int maxIssueStall = 0;
	for (unsigned int i = 0; i < MAX_NUM_WARP_PER_SM; i++)
	{
		unsigned int wId = (smId * MAX_NUM_WARP_PER_SM) + i;
		unsigned int issueStall = gWarpIssueStallArray[wId];
		printf("warp %u, issueStalls %u\n", wId, issueStall);
		if (issueStall > maxIssueStall)
		{
			maxIssueStall = issueStall;
			maxIssueStallWarpId = i;
		}
	}
	return maxIssueStallWarpId + 1;
}

void shader_core_ctx::init_warps( unsigned cta_id, unsigned start_thread, unsigned end_thread )
{
    address_type start_pc = next_pc(start_thread);
    if (m_config->model == POST_DOMINATOR) {
        unsigned start_warp = start_thread / m_config->warp_size;
        unsigned end_warp = end_thread / m_config->warp_size + ((end_thread % m_config->warp_size)? 1 : 0);
        for (unsigned i = start_warp; i < end_warp; ++i) {
            unsigned n_active=0;
            simt_mask_t active_threads;
            for (unsigned t = 0; t < m_config->warp_size; t++) {
                unsigned hwtid = i * m_config->warp_size + t;
                if ( hwtid < end_thread ) {
                    n_active++;
                    assert( !m_active_threads.test(hwtid) );
                    m_active_threads.set( hwtid );
                    active_threads.set(t);
                }
            }
            m_simt_stack[i]->launch(start_pc,active_threads);
            m_warp[i].init(start_pc,cta_id,i,active_threads, m_dynamic_warp_id);
            ++m_dynamic_warp_id;
            m_not_completed += n_active;
      }
      if (gNumWarpsPerBlock == 0)
      {
		gNumWarpsPerBlock = end_warp - start_warp;
        gNumTBsPerSM = kernel_max_cta_per_shader;
		gMaxNumResidentWarpsPerSm = gNumWarpsPerBlock * gNumTBsPerSM;
		gMaxNumResidentWarpsPerSched = (gMaxNumResidentWarpsPerSm / NUM_SCHED_PER_SM) + ((gMaxNumResidentWarpsPerSm % NUM_SCHED_PER_SM) ? 1 : 0);
        printf("Number of warps per block = %u\n", gNumWarpsPerBlock);
        printf("Number of blocks = %u\n", kernel_max_cta_per_shader);
        printf("Number of resident warps per SM = %u\n", gMaxNumResidentWarpsPerSm);
        printf("Max number of resident warps per sched = %u\n", gMaxNumResidentWarpsPerSched);
      }
	}

	valueMap* lQvalueTables[MAX_NUM_RL_ENGINES];
	valueUpdateMap* lQvalueUpdateTables[MAX_NUM_RL_ENGINES];
	SarsaAgent* lSarsaAgents[MAX_NUM_RL_ENGINES];
	for (unsigned int i = 0; i < MAX_NUM_RL_ENGINES; i++)
	{
		lQvalueTables[i] = 0;
		lQvalueUpdateTables[i] = 0;
		lSarsaAgents[i] = 0;
	}

    for (unsigned i = 0; i < schedulers.size(); i++)
	{
		if (schedulers[i]->isRLSched())
		{
			rl_scheduler* lRlSched = (rl_scheduler*)schedulers[i];

			for (unsigned int j = 0; j < lRlSched->dNumRLEngines; j++)
			{
				lRlSched->dRLEngines[j]->mInitTables(lQvalueTables[j], lQvalueUpdateTables[j], lSarsaAgents[j]);
			}
		}
	}
}

void rl_scheduler::mInitTables(valueMap*& xQvalueTable1, valueUpdateMap*& xQvalueUpdateTable1, 
							   valueMap*& xQvalueTable2, valueUpdateMap*& xQvalueUpdateTable2, 
							   SarsaAgent*& xSarsaAgent1, SarsaAgent*& xSarsaAgent2)
{
	assert(0);
	dRLEngines[0]->mInitTables(xQvalueTable1, xQvalueUpdateTable1, xSarsaAgent1);
	if (dRLEngines[1])
		dRLEngines[1]->mInitTables(xQvalueTable2, xQvalueUpdateTable2, xSarsaAgent2);
}

void rlEngine::mInitTables(valueMap*& xQvalueTable, valueUpdateMap*& xQvalueUpdateTable, SarsaAgent*& xSarsaAgent)
{
	dAttrString = dRLSched->dAttrString;
	if (gPrintFlag)
	{
        printf("Attributes %s kernel %s\n", dAttrString.c_str(), dRLSched->m_shader->get_kernel()->name().c_str());
		gPrintFlag = false;
	}
	if (dAttributeVector.size() > 0)
		return;

    this->addAttributes(dAttrString);

	if (gShareQvalueTableForAllSMs && (gQvalueTableForAllSMs[dEngineNum] != 0) && (gQvalueUpdateTableForAllSMs[dEngineNum] != 0))
	{
        this->allocateQvalues(gQvalueTableForAllSMs[dEngineNum]);
        this->allocateQvalueUpdates(gQvalueUpdateTableForAllSMs[dEngineNum]);

		if (rl_scheduler::gUseCMACFuncApprox)
		{
			assert(xSarsaAgent != 0);
			this->mCreateSarsaAgent(xSarsaAgent);
		}
	}
	else if (dRLSched->m_id != 0)
    {
        assert(xQvalueTable != 0);
        assert(xQvalueUpdateTable != 0);

        this->allocateQvalues(xQvalueTable);
        this->allocateQvalueUpdates(xQvalueUpdateTable);

		if (rl_scheduler::gUseCMACFuncApprox)
		{
			assert(xSarsaAgent != 0);
			this->mCreateSarsaAgent(xSarsaAgent);
		}
    }
    else
    {
		assert(xQvalueTable == 0);
		assert(xQvalueUpdateTable == 0);

        xQvalueTable = this->allocateQvalues(0);
        xQvalueUpdateTable = this->allocateQvalueUpdates(0);

		if (gShareQvalueTableForAllSMs == true)
		{
			gQvalueTableForAllSMs[dEngineNum] = xQvalueTable;
			gQvalueUpdateTableForAllSMs[dEngineNum] = xQvalueUpdateTable;
		}

        this->initQvalues();
        this->initQvalueUpdates();

		if (rl_scheduler::gUseCMACFuncApprox)
			xSarsaAgent = this->mCreateSarsaAgent(0);
    }

    this->initCurrStateAndAction();

}

// return the next pc of a thread 
address_type shader_core_ctx::next_pc( int tid ) const
{
    if( tid == -1 ) 
        return -1;
    ptx_thread_info *the_thread = m_thread[tid];
    if ( the_thread == NULL )
        return -1;
    return the_thread->get_pc(); // PC should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

void gpgpu_sim::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
    m_cluster[cluster_id]->get_pdom_stack_top_info(sid,tid,pc,rpc);
}

void shader_core_ctx::get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned warp_id = tid/m_config->warp_size;
    m_simt_stack[warp_id]->get_pdom_stack_top_info(pc,rpc);
}

unsigned int prevStallCnts_0 = 0;
unsigned int prevStallCnts_1 = 0;
unsigned int prevStallCnts_2 = 0;

#define DUMMY_ACTION     4

unsigned long long prev_gpgpu_n_load_insn = 0;
unsigned long long prev_gpgpu_n_store_insn = 0;
unsigned long long prev_gpgpu_n_shmem_insn = 0;
unsigned long long prev_gpgpu_n_tex_insn = 0;
unsigned long long prev_gpgpu_n_const_insn = 0;
unsigned long long prev_gpgpu_n_param_insn = 0;

void shader_core_stats::print( FILE* fout ) const
{
    unsigned long long  thread_icount_uarch=0;
    unsigned long long  warp_icount_uarch=0;

    for(unsigned i=0; i < m_config->num_shader(); i++) {
        thread_icount_uarch += m_num_sim_insn[i];
        warp_icount_uarch += m_num_sim_winsn[i];
    }
    fprintf(fout,"gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch);
    fprintf(fout,"gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);

    fprintf(fout,"gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem );
    fprintf(fout,"gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
    fprintf(fout,"gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
    fprintf(fout,"gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
    fprintf(fout,"gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
    fprintf(fout,"gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
    fprintf(fout,"gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

   fprintf(fout, "gpgpu_n_load_insn  = %llu(%llu)\n", gpgpu_n_load_insn, gpgpu_n_load_insn - prev_gpgpu_n_load_insn);
   fprintf(fout, "gpgpu_n_store_insn = %llu(%llu)\n", gpgpu_n_store_insn, gpgpu_n_store_insn - prev_gpgpu_n_store_insn);
   fprintf(fout, "gpgpu_n_shmem_insn = %llu(%llu)\n", gpgpu_n_shmem_insn, gpgpu_n_shmem_insn - prev_gpgpu_n_shmem_insn);
   fprintf(fout, "gpgpu_n_tex_insn = %llu(%llu)\n", gpgpu_n_tex_insn, gpgpu_n_tex_insn - prev_gpgpu_n_tex_insn);
   fprintf(fout, "gpgpu_n_const_mem_insn = %llu(%llu)\n", gpgpu_n_const_insn, gpgpu_n_const_insn - prev_gpgpu_n_const_insn);
   fprintf(fout, "gpgpu_n_param_mem_insn = %llu(%llu)\n", gpgpu_n_param_insn, gpgpu_n_param_insn - prev_gpgpu_n_param_insn);

	prev_gpgpu_n_load_insn = gpgpu_n_load_insn;
	prev_gpgpu_n_store_insn = gpgpu_n_store_insn;
	prev_gpgpu_n_shmem_insn = gpgpu_n_shmem_insn;
	prev_gpgpu_n_tex_insn = gpgpu_n_tex_insn;
	prev_gpgpu_n_const_insn = gpgpu_n_const_insn;
	prev_gpgpu_n_param_insn = gpgpu_n_param_insn;

   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
   fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);   

   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", gpgpu_n_intrawarp_mshr_merge);
   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][data_port_stall] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][DATA_PORT_STALL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][data_port_stall] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][DATA_PORT_STALL]);
   fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][bk_conf] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]   
           ); // coalescing stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][COAL_STALL]    
           ); // coalescing stall + bank conflict at data cache 

   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][DATA_PORT_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][DATA_PORT_STALL]    
           ); // data port stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

   fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n", gpu_reg_bank_conflict_stalls);

   fprintf(fout, "Warp Occupancy Distribution:\n");
   fprintf(fout, "Stall:%d(%u)\t", shader_cycle_distro[2], (shader_cycle_distro[2] - prevStallCnts_2));
   fprintf(fout, "W0_Idle:%d(%u)\t", shader_cycle_distro[0], (shader_cycle_distro[0] - prevStallCnts_0));
   fprintf(fout, "W0_Scoreboard:%d(%u)", shader_cycle_distro[1], (shader_cycle_distro[1] - prevStallCnts_1));
   for (unsigned i = 3; i < m_config->warp_size + 3; i++) 
      fprintf(fout, "\tW%d:%d", i-2, shader_cycle_distro[i]);
   fprintf(fout, "\n");
   fprintf(fout, "Mem pipeline stall:%u, Sfu pipeline stall:%u, sp pipeline stall:%u, mem & sfu pipeline stall:%u, mem & sp pipeline tall:%u, mem & sfu & sp pipeline stall:%u, sfu & sp pipeline stall:%u\n", gMemPipeLineStall, gSfuPipeLineStall, gSpPipeLineStall, gMemSfuPipeLineStall, gMemSpPipeLineStall, gMemSfuSpPipeLineStall, gSfuSpPipeLineStall);

	gMemPipeLineStall = 0;
	gSfuPipeLineStall = 0;
	gSpPipeLineStall = 0;
	gMemSfuSpPipeLineStall = 0;
	gMemSpPipeLineStall = 0;
	gMemSfuPipeLineStall = 0;
	gSfuSpPipeLineStall = 0;

   m_outgoing_traffic_stats->print(fout); 
   m_incoming_traffic_stats->print(fout); 

    if (gNumSMs != 0)
    {
           unsigned int actualStallCycles = ((shader_cycle_distro[2] + shader_cycle_distro[1] + shader_cycle_distro[0])  - (prevStallCnts_2 + prevStallCnts_1 + prevStallCnts_0)) / (gNumSMs * 2);
           fprintf(fout, "RL: Exploration Cycle Count = %u\n", rl_scheduler::gExplorationCnt / (gNumSMs * 2));
           fprintf(fout, "RL: Exploitation Cycle Count = %u\n", gExploitationCnt / (gNumSMs * 2));
           fprintf(fout, "RL: Actual Run Cycle Count = %llu\n", gpu_sim_cycle - actualStallCycles);

        gNumWarpsPerBlock = 0;
        rl_scheduler::gExplorationCnt = 0;
    }
   prevStallCnts_0 = shader_cycle_distro[0];
   prevStallCnts_1 = shader_cycle_distro[1];
   prevStallCnts_2 = shader_cycle_distro[2];
    rl_scheduler::gNumWarpsExecutingMemInstrGPU = 0;
    rl_scheduler::gNumReqsInMemSchedQs = 0;
    rl_scheduler::gNumMemSchedQsLoaded = 0;
    if (rl_scheduler::gGTCLongLatMemInstrCache)
    {
        for (unsigned int i = 0; i < GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE; i++)
            rl_scheduler::gGTCLongLatMemInstrCache[i] = 0;
    }
    if (rl_scheduler::gSFULongLatInstrCache)
    {
        for (unsigned int i = 0; i < SFU_LONG_LAT_INSTR_CACHE_SIZE; i++)
            rl_scheduler::gSFULongLatInstrCache[i] = 0;
    }
}

void shader_core_stats::event_warp_issued( unsigned s_id, unsigned warp_id, unsigned num_issued, unsigned dynamic_warp_id ) {
    assert( warp_id <= m_config->max_warps_per_shader );
    for ( unsigned i = 0; i < num_issued; ++i ) {
        if ( m_shader_dynamic_warp_issue_distro[ s_id ].size() <= dynamic_warp_id ) {
            m_shader_dynamic_warp_issue_distro[ s_id ].resize(dynamic_warp_id + 1);
        }
        ++m_shader_dynamic_warp_issue_distro[ s_id ][ dynamic_warp_id ];
        if ( m_shader_warp_slot_issue_distro[ s_id ].size() <= warp_id ) {
            m_shader_warp_slot_issue_distro[ s_id ].resize(warp_id + 1);
        }
        ++m_shader_warp_slot_issue_distro[ s_id ][ warp_id ];
    }
}

void shader_core_stats::visualizer_print( gzFile visualizer_file )
{
    // warp divergence breakdown
    gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
    unsigned int total=0;
    unsigned int cf = (m_config->gpgpu_warpdistro_shader==-1)?m_config->num_shader():1;
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf );
    for (unsigned i=0; i<m_config->warp_size+3; i++) {
       if ( i>=3 ) {
          total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
          if ( ((i-3) % (m_config->warp_size/8)) == ((m_config->warp_size/8)-1) ) {
             gzprintf(visualizer_file, " %d", total / cf );
             total=0;
          }
       }
       last_shader_cycle_distro[i] = shader_cycle_distro[i];
    }
    gzprintf(visualizer_file,"\n");

    // warp issue breakdown
    unsigned sid = m_config->gpgpu_warp_issue_shader;
    unsigned count = 0;
    unsigned warp_id_issued_sum = 0;
    gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
    if(m_shader_warp_slot_issue_distro[sid].size() > 0){
        for ( std::vector<unsigned>::const_iterator iter = m_shader_warp_slot_issue_distro[ sid ].begin();
              iter != m_shader_warp_slot_issue_distro[ sid ].end(); iter++, count++ ) {
            unsigned diff = count < m_last_shader_warp_slot_issue_distro.size() ?
                            *iter - m_last_shader_warp_slot_issue_distro[ count ] :
                            *iter;
            gzprintf( visualizer_file, " %d", diff );
            warp_id_issued_sum += diff;
        }
        m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[ sid ];
    }else{
        gzprintf( visualizer_file, " 0");
    }
    gzprintf(visualizer_file,"\n");

    #define DYNAMIC_WARP_PRINT_RESOLUTION 32
    unsigned total_issued_this_resolution = 0;
    unsigned dynamic_id_issued_sum = 0;
    count = 0;
    gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
    if(m_shader_dynamic_warp_issue_distro[sid].size() > 0){
        for ( std::vector<unsigned>::const_iterator iter = m_shader_dynamic_warp_issue_distro[ sid ].begin();
              iter != m_shader_dynamic_warp_issue_distro[ sid ].end(); iter++, count++ ) {
            unsigned diff = count < m_last_shader_dynamic_warp_issue_distro.size() ?
                            *iter - m_last_shader_dynamic_warp_issue_distro[ count ] :
                            *iter;
            total_issued_this_resolution += diff;
            if ( ( count + 1 ) % DYNAMIC_WARP_PRINT_RESOLUTION == 0 ) {
                gzprintf( visualizer_file, " %d", total_issued_this_resolution );
                dynamic_id_issued_sum += total_issued_this_resolution;
                total_issued_this_resolution = 0;
            }
        }
        if ( count % DYNAMIC_WARP_PRINT_RESOLUTION != 0 ) {
            gzprintf( visualizer_file, " %d", total_issued_this_resolution );
            dynamic_id_issued_sum += total_issued_this_resolution;
        }
        m_last_shader_dynamic_warp_issue_distro = m_shader_dynamic_warp_issue_distro[ sid ];
        assert( warp_id_issued_sum == dynamic_id_issued_sum );
    }else{
        gzprintf( visualizer_file, " 0");
    }
    gzprintf(visualizer_file,"\n");

    // overall cache miss rates
    gzprintf(visualizer_file, "gpgpu_n_cache_bkconflict: %d\n", gpgpu_n_cache_bkconflict);
    gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n", gpgpu_n_shmem_bkconflict);     


   // instruction count per shader core
   gzprintf(visualizer_file, "shaderinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_num_sim_insn[i] );
   gzprintf(visualizer_file, "\n");
   // warp instruction count per shader core
   gzprintf(visualizer_file, "shaderwarpinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++)
      gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i] );
   gzprintf(visualizer_file, "\n");
   // warp divergence per shader core
   gzprintf(visualizer_file, "shaderwarpdiv: ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_n_diverge[i] );
   gzprintf(visualizer_file, "\n");
}

#define PROGRAM_MEM_START 0xF0000000 /* should be distinct from other memory spaces... 
                                        check ptx_ir.h to verify this does not overlap 
                                        other memory spaces */
void shader_core_ctx::decode()
{
    if( m_inst_fetch_buffer.m_valid ) {
        // decode 1 or 2 instructions and place them into ibuffer
        address_type pc = m_inst_fetch_buffer.m_pc;
        const warp_inst_t* pI1 = ptx_fetch_inst(pc);
        m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(0,pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
        if( pI1 ) {
            m_stats->m_num_decoded_insn[m_sid]++;
            if(pI1->oprnd_type==INT_OP){
                m_stats->m_num_INTdecoded_insn[m_sid]++;
            }else if(pI1->oprnd_type==FP_OP) {
                m_stats->m_num_FPdecoded_insn[m_sid]++;
            }
               const warp_inst_t* pI2 = ptx_fetch_inst(pc+pI1->isize);
               if( pI2 ) {
                   m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(1,pI2);
                   m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
                   m_stats->m_num_decoded_insn[m_sid]++;
                   if(pI2->oprnd_type==INT_OP){
                       m_stats->m_num_INTdecoded_insn[m_sid]++;
                   }else if(pI2->oprnd_type==FP_OP) {
                       m_stats->m_num_FPdecoded_insn[m_sid]++;
                   }
                if (INSTR_BUFFER_SIZE > 2) {
                         const warp_inst_t* pI3 = ptx_fetch_inst(pc+pI1->isize+pI2->isize);
                       if( pI3 ) {
                           m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(2,pI3);
                           m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
                           m_stats->m_num_decoded_insn[m_sid]++;
                           if(pI3->oprnd_type==INT_OP){
                               m_stats->m_num_INTdecoded_insn[m_sid]++;
                           } else if(pI3->oprnd_type==FP_OP) {
                               m_stats->m_num_FPdecoded_insn[m_sid]++;
                           }
                        if (INSTR_BUFFER_SIZE > 3) {
                                 const warp_inst_t* pI4 = ptx_fetch_inst(pc+pI1->isize+pI2->isize+pI3->isize);
                               if( pI4 ) {
                                   m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(3,pI4);
                                   m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
                                   m_stats->m_num_decoded_insn[m_sid]++;
                                   if(pI4->oprnd_type==INT_OP){
                                       m_stats->m_num_INTdecoded_insn[m_sid]++;
                                   } else if(pI4->oprnd_type==FP_OP) {
                                       m_stats->m_num_FPdecoded_insn[m_sid]++;
                                   }
                               }
                        }
                       }
                }
               }
        }
        m_inst_fetch_buffer.m_valid = false;
    }
}

bool gWarpFinishing = false;
uint gStoreReqInProgress = 0;

void printLrrGtoCategoryRange(unsigned int smId, std::string kernelName)
{
	printf("Checking for all values of WOI:\n");
	for (unsigned int i = 0; i < gMaxNumResidentWarpsPerSm; i++)
	{
		int numWarpsOfInterest = i + 1;
		unsigned int totalScore = 0;
		unsigned int maxWarpScore = 0;
		for (int j = 0; j < numWarpsOfInterest; j++)
		{
			unsigned int wId = (smId * MAX_NUM_WARP_PER_SM) + j;
			unsigned int warpScore = gWarpProgressArray[wId] + gWarpBarrierTimeArray[wId];
			if (warpScore > maxWarpScore)
				maxWarpScore = warpScore;
			totalScore += warpScore;
		}
		unsigned int thresholdScore = numWarpsOfInterest * (maxWarpScore / 2);
		printf("thresholdScore = %u, totalScore = %u, numWarpsOfInterest = %u, maxWarpScore = %u, max num resident warps per sm = %u \n", thresholdScore, totalScore, numWarpsOfInterest, maxWarpScore, gMaxNumResidentWarpsPerSm);
		if (totalScore > thresholdScore)
		{
			printf("Using WOI = %u: LRR friendly kernel %s\n", numWarpsOfInterest, kernelName.c_str());
		}
		else
		{
			printf("Using WOI = %u: GTO friendly kernel %s\n", numWarpsOfInterest, kernelName.c_str());
		}
	}
}
void shader_core_ctx::fetch()
{
    //bool all_warps_functional_done = true;
    //bool all_warps_imiss_pending = true;
    //bool all_warps_ibuffer_empty = true;

    if( !m_inst_fetch_buffer.m_valid ) {
        // find an active warp with space in instruction buffer that is not already waiting on a cache miss
        // and get next 1-2 instructions from i-cache...
        for( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
            unsigned warp_id;
            warp_id = (m_last_warp_fetched+1+i) % m_config->max_warps_per_shader;
			/*
			uint smId = this->get_sid();
			if ((smId == 2) && (warp_id == 39) && gWarpFinishing)
			{
				static int printFlag = true;
				if (m_warp[warp_id].hardware_done() == true)
				{
					printf("%llu: warp %u, sm %u hardware done\n", gpu_sim_cycle, warp_id, smId);
					if (m_scoreboard->pendingWrites(warp_id) == false)
						printf("%llu: warp %u, sm %u pending writes done\n", gpu_sim_cycle, warp_id, smId);
					else
						printf("%llu: warp %u, sm %u pending writes NOT done\n", gpu_sim_cycle, warp_id, smId);
					printFlag = true;
				}
				else if (printFlag)
				{
					printFlag = false;
					printf("%llu: warp %u, sm %u hardware NOT done\n", gpu_sim_cycle, warp_id, smId);
					if (m_warp[warp_id].functional_done() == false)
						printf("%llu: warp %u, sm %u functional NOT done\n", gpu_sim_cycle, warp_id, smId);
					else if (m_warp[warp_id].stores_done() == false)
						printf("%llu: warp %u, sm %u stores NOT done\n", gpu_sim_cycle, warp_id, smId);
					else if (m_warp[warp_id].inst_in_pipeline() == true)
						printf("%llu: warp %u, sm %u pipeline NOT done\n", gpu_sim_cycle, warp_id, smId);
				}
			}
			*/

            // this code checks if this warp has finished executing and can be reclaimed
            if( m_warp[warp_id].hardware_done() && !m_scoreboard->pendingWrites(warp_id) && !m_warp[warp_id].done_exit() ) {
                bool did_exit=false;
                for( unsigned t=0; t<m_config->warp_size;t++) {
                    unsigned tid=warp_id*m_config->warp_size+t;
                    if( m_threadState[tid].m_active == true ) {
                        m_threadState[tid].m_active = false; 
                        unsigned cta_id = m_warp[warp_id].get_cta_id();

						if (did_exit == false) // do this only once
						{
                    		unsigned int smId = this->get_sid();
        					unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + warp_id;
							//printf("%llu: sm %u warp %u finished\n", gpu_sim_cycle, smId, warp_id);
							gWarpTimeArray[warpIdx] = gpu_sim_cycle - gWarpTimeArray[warpIdx];
							unsigned long long warpDrainTime = gpu_sim_cycle - gWarpDrainTimeArray[warpIdx];
							// printf("%llu: warp %u(%u), sm %u finished, drain time = %llu, num store reqs in progress = %u\n", gpu_sim_cycle, warp_id, m_warp[warp_id].get_dynamic_warp_id(), smId, gpu_sim_cycle - gWarpDrainTimeArray[warpIdx], gStoreReqInProgress);
							if (gMinDrainTime > warpDrainTime)
								gMinDrainTime = warpDrainTime;
							if (gMaxDrainTime < warpDrainTime)
								gMaxDrainTime = warpDrainTime;
							gTotalNumWarpsFinished++;
							if (gIPAWS)
							{
								if (gTotalNumWarpsFinished == 1)
								{
									bool lUseOracle = false;
									//lUseOracle = true;
									//For L1=16KB
									if ((m_kernel->name() == "_Z18executeFourthLayerPfS_S_") ||
									    (m_kernel->name() == "_Z24block2D_hybrid_coarsen_xffPfS_iii") ||
									    (m_kernel->name() == "_Z23mergeHistogram256KernelPjS_j") ||
									    (m_kernel->name() == "_Z18histogram256KernelPjS_j") ||
									    (m_kernel->name() == "_Z22mergeHistogram64KernelPjS_j") ||
									    (m_kernel->name() == "_Z27MonteCarloOneBlockPerOptionPfi") ||
									    (m_kernel->name() == "_Z13GPU_laplace3diiiiPfS_") ||
									    (m_kernel->name() == "_Z17cuda_compute_fluxiPiPfS0_S0_") ||
									    (m_kernel->name() == "_Z13lud_perimeterPfii") ||
									    (m_kernel->name() == "_Z12lud_diagonalPfii") ||
									    (m_kernel->name() == "_Z24cuda_compute_step_factoriPfS_S_") ||
									    (m_kernel->name() == "_Z11srad_cuda_2PfS_S_S_S_S_iiff") ||
									    (m_kernel->name() == "_Z17larger_sad_calc_8Ptii"))
									{
										gIPAWS_UseGTO = false; //LRR
										printf("Using Oracle: LRR friendly kernel %s\n", m_kernel->name().c_str());
									}
									else
									{
										gIPAWS_UseGTO = true; //GTO
										printf("Using Oracle: GTO friendly kernel %s\n", m_kernel->name().c_str());
									}
/*
									//For L1=32KB
									if (
									    (m_kernel->name() == "_Z13lud_perimeterPfii") ||
									    (m_kernel->name() == "_Z12lud_diagonalPfii") ||
									    (m_kernel->name() == "_Z17larger_sad_calc_8Ptii") ||
									    (m_kernel->name() == "_Z17larger_sad_calc_16Ptii") ||
									    (m_kernel->name() == "_Z15BlackScholesGPUPfS_S_S_S_ffi") ||
									    (m_kernel->name() == "_Z15mummergpuKernelP10MatchCoordPcPKiS3_ii") ||
									    (m_kernel->name() == "_Z27MonteCarloOneBlockPerOptionPfi") ||
									    (m_kernel->name() == "_Z22mergeHistogram64KernelPjS_j") ||
									    (m_kernel->name() == "_Z18histogram256KernelPjS_j") ||
									    (m_kernel->name() == "_Z23mergeHistogram256KernelPjS_j") ||
									    (m_kernel->name() == "_Z11srad_cuda_2PfS_S_S_S_S_iiff") ||
									    (m_kernel->name() == "_Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_S0_") ||
									    (m_kernel->name() == "_Z6KernelP4NodePiPbS2_S2_S1_i") ||
									    (m_kernel->name() == "_Z24block2D_hybrid_coarsen_xffPfS_iii") ||
									    (m_kernel->name() == "_Z17cuda_compute_fluxiPiPfS0_S0_") ||
									    (m_kernel->name() == "_Z13GPU_laplace3diiiiPfS_")
									    )
									{
										gIPAWS_UseGTO = false; //LRR
										printf("Using Oracle: LRR friendly kernel %s\n", m_kernel->name().c_str());
									}
									else
									{
										gIPAWS_UseGTO = true; //GTO
										printf("Using Oracle: GTO friendly kernel %s\n", m_kernel->name().c_str());
									}
*/

									//end of adapt phase
									if (lUseOracle == false)
									{
										unsigned int totalScore = 0;
										unsigned int numWarpsOfInterest = gGetNumWarpsOfInterest(smId);
										printf("Warps Of Interest = %u\n", numWarpsOfInterest);
										unsigned int maxWarpScore = 0;
										for (unsigned int i = 0; i < numWarpsOfInterest; i++)
										{
											unsigned int wId = (smId * MAX_NUM_WARP_PER_SM) + i;	
											unsigned int warpScore = gWarpProgressArray[wId] + gWarpBarrierTimeArray[wId];
											if (warpScore > maxWarpScore)
												maxWarpScore = warpScore;
											totalScore += warpScore;
										}
	
										unsigned int thresholdScore = numWarpsOfInterest * (maxWarpScore / 2);
										printf("thresholdScore = %u, totalScore = %u, numWarpsOfInterest = %u, maxWarpScore = %u, max num resident warps per sm = %u \n", thresholdScore, totalScore, numWarpsOfInterest, maxWarpScore, gMaxNumResidentWarpsPerSm);
										if (totalScore > thresholdScore)
										{
											gIPAWS_UseGTO = false;
											printf("Using WOI: LRR friendly kernel %s\n", m_kernel->name().c_str());
										}
										else
										{
											gIPAWS_UseGTO = true;
											printf("Using WOI: GTO friendly kernel %s\n", m_kernel->name().c_str());
										}


										printLrrGtoCategoryRange(smId, m_kernel->name());
									}

									if (gIPAWS_UseGTO == false)
									{
										gIPAWS_RecoverPhase = true;
										printf("%llu: Recover Phase started\n", gpu_sim_cycle);
										unsigned int minNumInstrs = 0xFFFFFFFF;
										for (unsigned int i = 0; i < gMaxNumResidentWarpsPerSm; i++)
										{
											unsigned int wId = (smId * MAX_NUM_WARP_PER_SM) + i;	
											unsigned int numInstrs = gWarpProgressArray[wId];
											if ((numInstrs != 0) && (numInstrs < minNumInstrs))
											{
												minNumInstrs = numInstrs;
												gSlowestWarpId = i;
											}
										}
										printf("Slowest warp is %u on sm %u\n", gSlowestWarpId, smId);
									}
									else
									{
										gIPAWS_UseGTO = true;	
									}
									gSmToFinishFirstWarp = smId;
								}
								if (gIPAWS_RecoverPhase && (gSmToFinishFirstWarp == smId))
								{
									if (warp_id == gSlowestWarpId)
									{
										gIPAWS_RecoverPhase = false;
										printf("%llu: Recover Phase finished\n", gpu_sim_cycle);
									}
								}
								gWarpBarrierTimeArray[warpIdx] = 0;
							}
							gTotalDrainTime += warpDrainTime;

							if ((smId == 2) && (warp_id == 39) && gWarpFinishing)
								gWarpFinishing = false;
	
        					unsigned int tbIdx = smId * MAX_NUM_TB_PER_SM + cta_id;
			
							if (gWarpTimeArray[warpIdx] > gTBMaxWarpTimeArray[tbIdx])
								gTBMaxWarpTimeArray[tbIdx] = gWarpTimeArray[warpIdx];

							if (gWarpTimeArray[warpIdx] < gTBMinWarpTimeArray[tbIdx])
								gTBMinWarpTimeArray[tbIdx] = gWarpTimeArray[warpIdx];

							gWarpTimeArray[warpIdx] = 0;
							gWarpDrainTimeArray[warpIdx] = 0;
						}

                        register_cta_thread_exit(cta_id);
                        m_not_completed -= 1;
                        m_active_threads.reset(tid);
                        assert( m_thread[tid]!= NULL );
                        did_exit=true;
                    }
                }
                if( did_exit ) 
                {
                    m_warp[warp_id].set_done_exit();
                    unsigned int smId = this->get_sid();
    				std::map<unsigned int, unsigned int>& splitWarpDynamicIdMap = gSplitWarpDynamicIdMapVec.at(smId);
					unsigned int dyn_warp_id = m_warp[warp_id].get_dynamic_warp_id();
					splitWarpDynamicIdMap.erase(dyn_warp_id);

                    assert(warp_id < MAX_NUM_WARP_PER_SM);
                    unsigned int index = smId * MAX_NUM_WARP_PER_SM + warp_id;
                    if (gWarpProgressArray)
                        gWarpProgressArray[index] = 0;

                    if (gSelectedWarp)
                    {
                        for (unsigned i = 0; i < schedulers.size(); i++) 
                        {
                            if (gSelectedWarp[smId * NUM_SCHED_PER_SM + i] == warp_id)
                            {
                                gSelectedWarp[smId * NUM_SCHED_PER_SM + i] = 0xdeaddead;
                                if (smId == 0)
                                    printf("%llu: high prio warp %u finished\n", gpu_sim_cycle, warp_id);
                            }
                        }
                    }
                }
            }


            // this code fetches instructions from the i-cache or generates memory requests
            if( !m_warp[warp_id].functional_done() && !m_warp[warp_id].imiss_pending() && m_warp[warp_id].ibuffer_empty() ) {
                address_type pc  = m_warp[warp_id].get_pc();
                address_type ppc = pc + PROGRAM_MEM_START;
                unsigned nbytes= (8 * INSTR_BUFFER_SIZE);

                unsigned offset_in_block = pc & (m_config->m_L1I_config.get_line_sz()-1);
                if( (offset_in_block+nbytes) > m_config->m_L1I_config.get_line_sz() )
                {
                    nbytes = (m_config->m_L1I_config.get_line_sz()-offset_in_block);
                }

                // TODO: replace with use of allocator
                // mem_fetch *mf = m_mem_fetch_allocator->alloc()
                mem_access_t acc(INST_ACC_R,ppc,nbytes,false);
                mem_fetch *mf = new mem_fetch(acc,
                                              NULL/*we don't have an instruction yet*/,
                                              READ_PACKET_SIZE,
                                              warp_id,
                                              m_sid,
                                              m_tpc,
                                              m_memory_config );
                std::list<cache_event> events;
                enum cache_request_status status = m_L1I->access( (new_addr_type)ppc, mf, gpu_sim_cycle+gpu_tot_sim_cycle,events);
                if( status == MISS ) {
                    m_last_warp_fetched=warp_id;
                    m_warp[warp_id].set_imiss_pending();
                    m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                } else if( status == HIT ) {
                    m_last_warp_fetched=warp_id;
                    m_inst_fetch_buffer = ifetch_buffer_t(pc,nbytes,warp_id);
                    m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                    delete mf;
                } else {
                    m_last_warp_fetched=warp_id;
                    assert( status == RESERVATION_FAIL );
                    delete mf;
                }
                if ((unsigned int)m_last_warp_fetched == smallerIssuedWarpId0)
                    smallerIssuedWarpId0 = 0xdeaddead;
                else if ((unsigned int)m_last_warp_fetched == smallerIssuedWarpId1)
                    smallerIssuedWarpId1 = 0xdeaddead;

                //if ((this->m_sid == 0) && (schedulers[0]->isRLSched() == 0))
                    //printf("FETCH_INSTR: warp_dynamic_id %u cta_id %u at cycle %llu, (%s)\n", m_warp[warp_id].get_dynamic_warp_id(), m_warp[warp_id].get_cta_id(), gpu_sim_cycle, (status == MISS) ? "MISS" : (status == HIT) ? "HIT" : "RESERVATION_FAIL");

                break;
            }
        }
    }

    m_L1I->cycle();

    if( m_L1I->access_ready() ) {
        mem_fetch *mf = m_L1I->next_access();
        m_warp[mf->get_wid()].clear_imiss_pending();
        delete mf;
    }
}

void shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
    execute_warp_inst_t(inst);
    if( inst.is_load() || inst.is_store() )
        inst.generate_mem_accesses();
}

void shader_core_ctx::issue_warp( register_set& pipe_reg_set, const warp_inst_t* next_inst, const active_mask_t &active_mask, unsigned warp_id )
{
    warp_inst_t** pipe_reg = pipe_reg_set.get_free();
    assert(pipe_reg);
    
    m_warp[warp_id].ibuffer_free();
    assert(next_inst->valid());
    **pipe_reg = *next_inst; // static instruction information
    (*pipe_reg)->issue( active_mask, warp_id, gpu_tot_sim_cycle + gpu_sim_cycle, m_warp[warp_id].get_dynamic_warp_id() ); // dynamic instruction information
    m_stats->shader_cycle_distro[2+(*pipe_reg)->active_count()]++;
    func_exec_inst( **pipe_reg );
    if( next_inst->op == BARRIER_OP ) 
        m_barriers.warp_reaches_barrier(m_warp[warp_id].get_cta_id(),warp_id);
    else if( next_inst->op == MEMORY_BARRIER_OP ) 
        m_warp[warp_id].set_membar();

    updateSIMTStack(warp_id,*pipe_reg);
    m_scoreboard->reserveRegisters(*pipe_reg);
    m_warp[warp_id].set_next_pc(next_inst->pc + next_inst->isize);
}

#define SCHED_PROG_DIFF_THRESHOLD 20

unsigned int gNumCellsTouchedPrimary = 0;
unsigned int gNumCellsTouchedSecondary = 0;

void rlEngine::printQValStat()
{
	unsigned int lNumCellsTouched;
	unsigned int lNumUpdates = mGetTotalQvalueUpdates(lNumCellsTouched);
	printf("Engine %u: total q value = %e total num of updates = %u num of cells touched = %u new cells touched = %u\n", 
			dEngineNum, mGetTotalQvalue(), lNumUpdates,lNumCellsTouched, 
			(lNumCellsTouched - gNumCellsTouchedPrimary));
}

bool gPrintNoMoreCTAsMsg = true;
void shader_core_ctx::issue(){
    //really is issue;

    bool moreTBsLeft = m_cluster->get_gpu()->get_more_cta_left() ? true : false;
    if ((moreTBsLeft == false) && gPrintNoMoreCTAsMsg)
    {
        printf("%llu: All TBs assigned to SMs\n", gpu_sim_cycle);
        gPrintNoMoreCTAsMsg = false;
		{
			rl_scheduler* schedPtr = ((rl_scheduler*)schedulers[0]);
			if (schedPtr->isRLSched())
			{
				for (unsigned int i = 0; i < schedPtr->dNumRLEngines; i++)
				{
					schedPtr->dRLEngines[i]->printQValStat();
				}
			}
		}
    }

    smallerIssuedWarpId0 = 0xdeaddead;
    smallerIssuedWarpId1 = 0xdeaddead;
    {
        for (unsigned i = 0; i < schedulers.size(); i++) {
            schedulers[i]->cycle();
        }
    }
}

shd_warp_t& scheduler_unit::warp(int i){
    return (*m_warp)[i];
}


/**
 * A general function to order things in a Loose Round Robin way. The simplist use of this
 * function would be to implement a loose RR scheduler between all the warps assigned to this core.
 * A more sophisticated usage would be to order a set of "fetch groups" in a RR fashion.
 * In the first case, the templated class variable would be a simple unsigned int representing the
 * warp_id.  In the 2lvl case, T could be a struct or a list representing a set of warp_ids.
 * @param result_list: The resultant list the caller wants returned.  This list is cleared and then populated
 *                     in a loose round robin way
 * @param input_list: The list of things that should be put into the result_list. For a simple scheduler
 *                    this can simply be the m_supervised_warps list.
 * @param last_issued_from_input:  An iterator pointing the last member in the input_list that issued.
 *                                 Since this function orders in a RR fashion, the object pointed
 *                                 to by this iterator will be last in the prioritization list
 * @param num_warps_to_add: The number of warps you want the scheudler to pick between this cycle.
 *                          Normally, this will be all the warps availible on the core, i.e.
 *                          m_supervised_warps.size(). However, a more sophisticated scheduler may wish to
 *                          limit this number. If the number if < m_supervised_warps.size(), then only
 *                          the warps with highest RR priority will be placed in the result_list.
 */
template < class T >
void scheduler_unit::order_lrr( std::vector< T >& result_list,
                                const typename std::vector< T >& input_list,
                                const typename std::vector< T >::const_iterator& last_issued_from_input,
                                unsigned num_warps_to_add )
{
    assert( num_warps_to_add <= input_list.size() );
    result_list.clear();
    typename std::vector< T >::const_iterator iter
        = ( last_issued_from_input ==  input_list.end() ) ? input_list.begin()
                                                          : last_issued_from_input + 1;

    for ( unsigned count = 0;
          count < num_warps_to_add;
          ++iter, ++count) {
        if ( iter ==  input_list.end() ) {
            iter = input_list.begin();
        }
        result_list.push_back( *iter );
    }
}


/**
 * A general function to order things in an priority-based way.
 * The core usage of the function is similar to order_lrr.
 * The explanation of the additional parameters (beyond order_lrr) explains the further extensions.
 * @param ordering: An enum that determines how the age function will be treated in prioritization
 *                  see the definition of OrderingType.
 * @param priority_function: This function is used to sort the input_list.  It is passed to stl::sort as
 *                           the sorting fucntion. So, if you wanted to sort a list of integer warp_ids
 *                           with the oldest warps having the most priority, then the priority_function
 *                           would compare the age of the two warps.
 */
template < class T >
void scheduler_unit::order_by_priority( std::vector< T >& result_list,
                                        const typename std::vector< T >& input_list,
                                        const typename std::vector< T >::const_iterator& last_issued_from_input,
                                        unsigned num_warps_to_add,
                                        OrderingType ordering,
                                        bool (*priority_func)(T lhs, T rhs) )
{
    assert( num_warps_to_add <= input_list.size() );
    result_list.clear();
    typename std::vector< T > temp = input_list;

    if ( ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering ) {
        T greedy_value = *last_issued_from_input;
        result_list.push_back( greedy_value );

        std::sort( temp.begin(), temp.end(), priority_func );
        typename std::vector< T >::iterator iter = temp.begin();
        for ( unsigned count = 0; count < num_warps_to_add; ++count, ++iter ) {
            if ( *iter != greedy_value ) {
                result_list.push_back( *iter );
            }
        }
    } else if ( ORDERING_RANDOM_GREEDY_THEN_PRIORITY_FUNC == ordering ) {
        long int randVal = random();
		float randomPercent = (((float)randVal/(float)RAND_MAX)) * 100;
		if (randomPercent < gRTOSchedRandomPercent)
		{
			gRTOSchedRandomOrderCnt++;
        	long int randVal = random();
        	unsigned int idx = randVal % temp.size();
        	shd_warp_t* warp = temp[idx];
        	result_list.push_back(warp);
    	
        	for (unsigned i = (idx + 1); i < temp.size(); ++i)
            	result_list.push_back(temp[i]);
    	
        	for (unsigned i = 0; i < idx; ++i)
            	result_list.push_back(temp[i]);
		}
		else
		{
			gRTOSchedGTOOrderCnt++;
        	T greedy_value = *last_issued_from_input;
        	result_list.push_back( greedy_value );

        	std::sort( temp.begin(), temp.end(), priority_func );
        	typename std::vector< T >::iterator iter = temp.begin();
        	for ( unsigned count = 0; count < num_warps_to_add; ++count, ++iter ) {
            	if ( *iter != greedy_value ) {
                	result_list.push_back( *iter );
            	}
        	}
		}
    } else if ( ORDERED_PRIORITY_FUNC_ONLY == ordering ) {
        std::sort( temp.begin(), temp.end(), priority_func );
        typename std::vector< T >::iterator iter = temp.begin();
        for ( unsigned count = 0; count < num_warps_to_add; ++count, ++iter ) {
            result_list.push_back( *iter );
        }
    } else {
        fprintf( stderr, "Unknown ordering - %d\n", ordering );
        abort();
    }
}


shd_warp_t* ipaws_scheduler::getOldestReadyWarp(std::vector<shd_warp_t*>& xStalledWarpVec)
{
    std::vector<shd_warp_t*> warpVec;
    order_by_priority( warpVec,
                      	m_supervised_warps,
                      	m_last_supervised_issued,
                      	m_supervised_warps.size(),
                      	ORDERED_PRIORITY_FUNC_ONLY,
                      	scheduler_unit::sort_warps_by_oldest_dynamic_id );

	shd_warp_t* returnWarpPtr = getFirstReadyWarp(warpVec, xStalledWarpVec);
	return returnWarpPtr;
}

shd_warp_t* scheduler_unit::getFirstReadyWarp(std::vector<shd_warp_t*>& xSortedWarpVec, std::vector<shd_warp_t*>& xStalledWarpVec)
{
	shd_warp_t* lFirstRdyWarp = 0;
    for (std::vector<shd_warp_t*>::const_iterator iter = xSortedWarpVec.begin(); 
         iter != xSortedWarpVec.end(); 
         iter++) 
    {
		shd_warp_t* warpPtr = (*iter);
        // Don't consider warps that are not yet valid
        if ((*iter) == NULL)
               continue;
        if ((*iter)->done_exit())
            continue;

        unsigned warp_id = (*iter)->get_warp_id();
		bool lIsWarpRdy = false;

        if (!warp(warp_id).ibuffer_empty())
        {
            if (m_shader->warp_waiting_at_barrier(warp_id) == false)
			{
            	const warp_inst_t* pI1 = warp(warp_id).ibuffer_next_inst();
            	if(pI1) 
            	{
                	unsigned pc, rpc;
	
                	m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc, &rpc);
	
                	if( pc == pI1->pc ) 
                	{
                    	if (!m_scoreboard->checkCollision(warp_id, pI1)) 
                    	{
                        	if ((pI1->op == LOAD_OP) || (pI1->op == STORE_OP) || (pI1->op == MEMORY_BARRIER_OP)) 
                        	{
                            	if(m_mem_out->has_free()) 
								{
									lIsWarpRdy = true;
									if (lFirstRdyWarp == 0)
                                		lFirstRdyWarp = warpPtr;
								}
                        	}
                        	else 
                        	{
                            	bool sp_pipe_avail = m_sp_out->has_free();
                            	bool sfu_pipe_avail = m_sfu_out->has_free();
								if (sp_pipe_avail && (pI1->op != SFU_OP))
								{
									lIsWarpRdy = true;
                                	if (lFirstRdyWarp == 0)
										lFirstRdyWarp = warpPtr;
								}
								else if (sfu_pipe_avail && ((pI1->op == SFU_OP) || (pI1->op == ALU_SFU_OP)))
								{
									lIsWarpRdy = true;
                                	if (lFirstRdyWarp == 0)
										lFirstRdyWarp = warpPtr;
								}
                        	}
                    	}
                	}
            	} 
			}
        }
		if (lIsWarpRdy == false)
			xStalledWarpVec.push_back(warpPtr);
	}
	return lFirstRdyWarp;
}

shd_warp_t* rl_scheduler::getGTOWarp()
{
	std::vector<shd_warp_t*> lStalledWarpVec;
	shd_warp_t* returnWarpPtr = getFirstReadyWarp(gGTOWarpOrder, lStalledWarpVec);
	return returnWarpPtr;
}

//std::map<unsigned long long, unsigned int> gCycleWarpMap;
//std::map<unsigned long long, unsigned int> gCycleCmdMap;

void gSetRewardAndPenalty(bool valid_inst, bool ready_inst, operation_pipeline_t pipeUsed)
{
    if (gDiffPenalty == 1)
    {
        if (!valid_inst)
            gPenalty = -2;
        else if (!ready_inst)
            gPenalty = -1;
        else
            gPenalty = 0;
    }
    else if (gDiffPenalty == 2)
    {
        if (!valid_inst)
            gPenalty = -4;
        else if (!ready_inst)
            gPenalty = -2;
        else
            gPenalty = 0;
    }
    else if (gDiffPenalty == 3)
    {
        if (!valid_inst)
            gPenalty = -10;
        else if (!ready_inst)
            gPenalty = -5;
        else
            gPenalty = -1;
    }
    else if (gDiffPenalty == 4)
    {
        if (!valid_inst)
            gPenalty = -16;
        else if (!ready_inst)
            gPenalty = -8;
        else
            gPenalty = -2;
    }
    else if (gDiffPenalty == 5)
        gPenalty = -1;

    if (gDiffReward == 1)
    {
        if (pipeUsed == MEM__OP)
            gReward = 3;
        else if (pipeUsed == SFU__OP)
            gReward = 2;
        else if (pipeUsed == SP__OP)
            gReward = 1;
    }
    else if (gDiffReward == 2)
    {
        if (pipeUsed == MEM__OP)
            gReward = 5;
        else if (pipeUsed == SFU__OP)
            gReward = 3;
        else if (pipeUsed == SP__OP)
            gReward = 1;
    }
    else if (gDiffReward == 3)
    {
        if (pipeUsed == MEM__OP)
            gReward = 10;
        else if (pipeUsed == SFU__OP)
            gReward = 5;
        else if (pipeUsed == SP__OP)
            gReward = 1;
    }
    else if (gDiffReward == 4)
    {
        if (pipeUsed == MEM__OP)
            gReward = 16;
        else if (pipeUsed == SFU__OP)
            gReward = 8;
        else if (pipeUsed == SP__OP)
            gReward = 2;
    }
    else if (gDiffReward == 5)
    {
        gReward = 1;
    }
}

extern time_t g_simulation_starttime;
#define ONE_HOUR 3600
#define BM_RUNTIME_THRESHOLD (1 * 24 * ONE_HOUR)
bool runningLongerThanThreshold()
{
    time_t current_time = time((time_t *)NULL);
    time_t difference = MAX(current_time - g_simulation_starttime, 1);
    if (difference > BM_RUNTIME_THRESHOLD)
        return true;
    return false;
}


void scheduler_unit::cycle()
{
   if (((gpu_tot_sim_cycle + gpu_sim_cycle) % 1000) == 0)
    {
        bool longRunning = runningLongerThanThreshold();
        if (longRunning)
        {
            printf("Running longer than threshold, killing, completed %llu cycles\n", (gpu_tot_sim_cycle + gpu_sim_cycle));
            exit(0);
        }
    }

    uint smId = m_shader->get_sid();
    uint schedId = this->m_id;
    char instrTypeStr[10];
    strcpy(instrTypeStr, "NOP");

    operation_pipeline_t pipeUsed = UNKNOWN_OP;
    warp_inst_t* instrSched = 0;
    unsigned int activeMaskCount = 0;

    SCHED_DPRINTF( "scheduler_unit::cycle()\n" );
    bool valid_inst = false;  // there was one warp with a valid instruction to issue (didn't require flush due to control hazard)
    bool ready_inst = false;  // of the valid instructions, there was one not waiting for pending register writes
    bool issued_inst = false; // of these we issued one
    shd_warp_t* issued_warp = 0;
    bool sp_op = false;
    bool sfu_op = false;
    bool mem_op = false;

    order_warps();

	shd_warp_t* gtoWarp = ((rl_scheduler*)this)->getGTOWarp();

/*
	if (gIPAWS)
	{
		shd_warp_t* oldestRdyWarp = 0;
		std::vector<shd_warp_t*> lStalledWarpVec;
		oldestRdyWarp = ((ipaws_scheduler*)this)->getOldestReadyWarp(lStalledWarpVec);
		for (unsigned int i = 0; i < lStalledWarpVec.size(); i++)
		{
			shd_warp_t* lStalledWarp = lStalledWarpVec[i];
        	unsigned int warpId = lStalledWarp->get_warp_id();
        	unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + warpId;

        	if (gWarpIssueStallArray)
            	gWarpIssueStallArray[warpIdx]++;
		}
	}
*/

	gCnt0++;
	unsigned int notIssuedWarpId = 0xdeaddead;

    for ( std::vector< shd_warp_t* >::const_iterator iter = m_next_cycle_prioritized_warps.begin();
          iter != m_next_cycle_prioritized_warps.end();
          iter++ ) {
	
		gCnt1++;
        // Don't consider warps that are not yet valid
        if ( (*iter) == NULL ) {
            continue;
        }

		gCnt1_5++;
        if ( (*iter)->done_exit() ) {
            continue;
        }

		if (notIssuedWarpId != 0xdeaddead)
		{
       		unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + notIssuedWarpId;

        	if (gWarpIssueStallArray)
            	gWarpIssueStallArray[warpIdx]++;
		}
        unsigned warp_id = (*iter)->get_warp_id();
		notIssuedWarpId = warp_id;

		gCnt2++;

        SCHED_DPRINTF( "Testing (warp_id %u, dynamic_warp_id %u)\n",
                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
        unsigned checked=0;
        unsigned issued=0;
        unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;
		if (warp(warp_id).waiting())
			gCnt2_1++;
		else if (warp(warp_id).ibuffer_empty())
			gCnt2_2++;
		else if (checked >= max_issue)
			gCnt2_3++;
		else if (checked > issued)
			gCnt2_4++;
		else if (issued >= max_issue)
			gCnt2_5++;

        while( !warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() && (checked < max_issue) && (checked <= issued) && (issued < max_issue) ) {

			gCnt3++;

            const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
            bool valid = warp(warp_id).ibuffer_next_valid();
            bool warp_inst_issued = false;
            unsigned pc,rpc;
            m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
                           (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
                           ptx_get_insn_str( pc).c_str() );
            if( pI ) {

				gCnt4++;

                assert(valid);
                if( pc != pI->pc ) {
					gCnt5++;
                    SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) control hazard instruction flush\n",
                                   (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                    // control hazard
                    warp(warp_id).set_next_pc(pc);
                    warp(warp_id).ibuffer_flush();
                } else {
                    valid_inst = true;
					gCnt6++;
                    if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                        SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                        ready_inst = true;
						gCnt7++;
                        const active_mask_t &active_mask = m_simt_stack[warp_id]->get_active_mask();
                        assert( warp(warp_id).inst_in_pipeline() );
                        if (((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP)) && 
                             (rl_scheduler::gSelectedActionVal != UNKNOWN_OP)  ) 
                        {
                            bool skipInstr = false;

                            mem_op = true;
                            if( (skipInstr == false) && m_mem_out->has_free() ) {
                                m_shader->issue_warp(*m_mem_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
								gCnt8++;
                                issued_warp = (*iter);
                                instrSched = (warp_inst_t*)pI;

                                warp_inst_issued = true;
                                strcpy(instrTypeStr, "MEM");
                                pipeUsed = MEM__OP;
                                activeMaskCount = active_mask.count();
								if (gIsHighPrioMemReq)
									instrSched->mSetHighPrioInst();
								else
									instrSched->mResetHighPrioInst();
								if (gBypassL1Cache)
									instrSched->mSetBypassL1Cache();
								else
									instrSched->mResetBypassL1Cache();
                            }
                        } else {
                             if ( (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP) )
                                sfu_op = true;
                            else
                                sp_op = true;

                            bool sp_pipe_avail = m_sp_out->has_free();
                            bool sfu_pipe_avail = m_sfu_out->has_free();
                            if( sp_pipe_avail && (pI->op != SFU_OP) && (rl_scheduler::gSelectedActionVal != UNKNOWN_OP)) {
                                // always prefer SP pipe for operations that can use both SP and SFU pipelines
                                m_shader->issue_warp(*m_sp_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
								gCnt8++;
                                strcpy(instrTypeStr, "SP");
								if (this->isRLSched())
								{
									rl_scheduler* rlSched = (rl_scheduler*)this;
									assert(rlSched->numSpPipeStalls == 0);
								}
                                pipeUsed = SP__OP;
                                instrSched = (warp_inst_t*)pI;
                                activeMaskCount = active_mask.count();
                                issued_warp = (*iter);
                                warp_inst_issued = true;

                            } else if (( (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) && (rl_scheduler::gSelectedActionVal != UNKNOWN_OP) ) {
                                if( sfu_pipe_avail ) {
                                    m_shader->issue_warp(*m_sfu_out,pI,active_mask,warp_id);
                                    issued++;
                                    issued_inst=true;
									gCnt8++;
                                    strcpy(instrTypeStr, "SFU");
                                    pipeUsed = SFU__OP;
                                    instrSched = (warp_inst_t*)pI;
                                    activeMaskCount = active_mask.count();
                                    issued_warp = (*iter);
                                    warp_inst_issued = true;

                                }
                            } 
                        }
                    } else {
                        SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                                       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
                    }
                }
            } else if( valid ) {
               // this case can happen after a return instruction in diverged warp
               SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp flush\n",
                              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id() );
               warp(warp_id).set_next_pc(pc);
               warp(warp_id).ibuffer_flush();
            }
            if(warp_inst_issued) {
                SCHED_DPRINTF( "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
                               (*iter)->get_warp_id(),
                               (*iter)->get_dynamic_warp_id(),
                               issued );
                do_on_warp_issued( warp_id, issued, iter );
            }
            checked++;
        }
        if ( issued ) {
            // This might be a bit inefficient, but we need to maintain
            // two ordered list for proper scheduler execution.
            // We could remove the need for this loop by associating a
            // supervised_is index with each entry in the m_next_cycle_prioritized_warps
            // vector. For now, just run through until you find the right warp_id
            for ( std::vector< shd_warp_t* >::const_iterator supervised_iter = m_supervised_warps.begin();
                  supervised_iter != m_supervised_warps.end();
                  ++supervised_iter ) {
                if ( *iter == *supervised_iter ) {
					{
                    	m_last_supervised_issued = supervised_iter;
					}
                }
            }
			notIssuedWarpId = 0xdeaddead;
            break;
        } 
    }
	if (notIssuedWarpId != 0xdeaddead)
	{
      	unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + notIssuedWarpId;

       	if (gWarpIssueStallArray)
           	gWarpIssueStallArray[warpIdx]++;
	}

	/*
	if (instrSched)
	{
    	std::map<unsigned int, unsigned int>& splitWarpDynamicIdMap = gSplitWarpDynamicIdMapVec.at(smId);
		unsigned int dyn_warp_id = issued_warp->get_dynamic_warp_id();
		if (splitWarpDynamicIdMap.find(dyn_warp_id) == splitWarpDynamicIdMap.end())
			splitWarpDynamicIdMap[dyn_warp_id] = activeMaskCount;
	}
	*/

	//collect stall run cycles
	if (gpu_sim_cycle == 1)
	{
		if (issued_warp)
		{
			lastCycleRunCycle = true;
			firstCycleRunCycle = true;
		}
		else
		{
			lastCycleRunCycle = false;
			firstCycleRunCycle = false;
		}
		consCycleCnt = 1;
	}
	else
	{
		if (issued_warp)
		{
			//if ((smId == 0) && (schedId == 0))
			//{
				//gCycleWarpMap[gpu_sim_cycle] = issued_warp->get_dynamic_warp_id();
			//}

			lastSchedRunCycle = gpu_sim_cycle;
			if (lastCycleRunCycle)
				consCycleCnt++;
			else
			{
				if (stallCycleCntMap.find(consCycleCnt) == stallCycleCntMap.end())
					stallCycleCntMap[consCycleCnt] = 1;
				else
					stallCycleCntMap[consCycleCnt]++;

				runStallCyclesVec.push_back(consCycleCnt);
				consCycleCnt = 1;
				lastCycleRunCycle = true;
			}
		}
		else
		{
			if (lastCycleRunCycle)
			{
				if (runCycleCntMap.find(consCycleCnt) == runCycleCntMap.end())
					runCycleCntMap[consCycleCnt] = 1;
				else
					runCycleCntMap[consCycleCnt]++;

				runStallCyclesVec.push_back(consCycleCnt);
				consCycleCnt = 1;
				lastCycleRunCycle = false;
			}
			else
				consCycleCnt++;

		}
	}

	/*
	if (issued_warp)
	{
    	if (smId == 2)
		{
			if (issued_warp->get_warp_id() == 39)
			{
        		std::string instrStr = ptx_get_insn_str(instrSched->pc);
				printf("%llu: issued instr %s by warp %u on sm %u\n", gpu_sim_cycle, instrStr.c_str(), issued_warp->get_warp_id() , smId);
			}
		}
	}
	*/

    // issue stall statistics:
    if( !valid_inst ) 
    {
        m_stats->shader_cycle_distro[0]++; // idle or control hazard
		schedStateVec.push_back(IDLE_STALL);
    }
    else if( !ready_inst ) 
    {
        m_stats->shader_cycle_distro[1]++; // waiting for RAW hazards (possibly due to memory) 
		schedStateVec.push_back(SB_STALL);
    }
    else if( !issued_inst ) 
    {
        m_stats->shader_cycle_distro[2]++; // pipeline stalled
        if (mem_op)
            gMemPipeLineStall++;
        if (sfu_op)
            gSfuPipeLineStall++;
        if (sp_op)
            gSpPipeLineStall++;
        if (mem_op && sfu_op)
            gMemSfuPipeLineStall++;
        if (mem_op && sp_op)
            gMemSpPipeLineStall++;
        if (sfu_op && sp_op)
            gSfuSpPipeLineStall++;
        if (mem_op && sfu_op && sp_op)
            gMemSfuSpPipeLineStall++;
		schedStateVec.push_back(PIPE_STALL);
    }
	else
		schedStateVec.push_back(NO_STALL);

    if (gPrintDRAMInfo && (pipeUsed == MEM__OP) &&(smId == 0))
    {
        if (gNumReqsInMemSchedArray)
        {
            printf("NMIE=%u, cycle=%llu", rl_scheduler::gNumWarpsExecutingMemInstrGPU, gpu_sim_cycle);
            for (unsigned int i = 0; i < 6; i++)
                printf(",%u", gNumReqsInMemSchedArray[i]);
            printf("\n");
        }
    }

    //For RL scheduler
    if (isRLSched())
    {
		if (issued_warp)
		{
			rl_scheduler* rlSched = (rl_scheduler*)this;
			rlSched->dFirstWarpIssued = true;
			unsigned int nonEmptyInstrTypes = 0;
			if (rlSched->numReadySpInstrs > 0)
				nonEmptyInstrTypes++;
			if (rlSched->numReadySfuInstrs > 0)
				nonEmptyInstrTypes++;
			if (rlSched->numReadyMemInstrs> 0) 
			{
				nonEmptyInstrTypes++;
				if (rlSched->numReadySharedTexConstMemInstrs > 0)
					nonEmptyInstrTypes++;
			}

			if ((rlSched->numReadyInstrs > 1) && (nonEmptyInstrTypes > 1))
			{
				assert(gtoWarp);
				if (issued_warp == gtoWarp)
					gSameWarpAsGTOCnt++;
				else
					gNotSameWarpAsGTOCnt++;
			}
		}

        if (((rl_scheduler*)this)->dFirstWarpIssued && m_shader->isactive())
        {
            if (issued_warp && (gReadyTBIdSet.size() == 0))
                printf("%llu: smId = %u, schedId = %u no ready TBs but issued warp %u(%u)\n", gpu_sim_cycle, smId, schedId, issued_warp->get_warp_id(), issued_warp->get_cta_id());

            computeNextValueAndUpdateOldValue(issued_warp, pipeUsed, instrSched);

			gSetRewardAndPenalty(valid_inst, ready_inst, pipeUsed);

            collectRewardAndSetAttributes(issued_inst, issued_warp, pipeUsed, instrTypeStr, instrSched);
        }
    }

    if (issued_warp)
    {
		if (rl_scheduler::gNumInstrsIssued)
			rl_scheduler::gNumInstrsIssued[smId]++;
		if (pipeUsed == MEM__OP)
		{
            rl_scheduler::gNumWarpsExecutingMemInstrGPU++;
            if (rl_scheduler::gNumWarpsExecutingMemInstr)
                rl_scheduler::gNumWarpsExecutingMemInstr[smId]++;

            if ((instrSched->space.get_type() == global_space) || 
                (instrSched->space.get_type() == const_space) ||
                (instrSched->space.get_type() == tex_space))
            {
                rl_scheduler::gNumGTCMemInstrIssued++;
                if (gpu_sim_cycle > 50)
                    rl_scheduler::gNumGTCMemInstrIssued1++;
            }
            if (rl_scheduler::gLastMemInstrTB)
                rl_scheduler::gLastMemInstrTB[smId] = issued_warp->get_cta_id();
            if (rl_scheduler::gLastMemInstrPC)
                rl_scheduler::gLastMemInstrPC[smId] = instrSched->pc;
		}
		if (pipeUsed == SP__OP)
		{
            rl_scheduler::gNumSpInstrIssued++;
            if (gpu_sim_cycle > 50)
                rl_scheduler::gNumSpInstrIssued1++;
		}
		if (pipeUsed == SFU__OP)
		{
            rl_scheduler::gNumSfuInstrIssued++;
            if (gpu_sim_cycle > 50)
                rl_scheduler::gNumSfuInstrIssued1++;
		}

        //this is for the first warp that executes on each sched
        if (gSelectedTB && (gSelectedTB[smId] == 0xdeaddead))
        {
            gSelectedTB[smId] = issued_warp->get_cta_id();
            if (smId == 0)
                printf("%llu: Selected %u as high prio tb (first)\n", gpu_sim_cycle, issued_warp->get_cta_id());
        }
        if (gSelectedWarp && (gSelectedWarp[smId * NUM_SCHED_PER_SM + schedId] == 0xdeaddead))
        {
            if (issued_warp->get_cta_id() == gSelectedTB[smId])
            {
                gSelectedWarp[smId * NUM_SCHED_PER_SM + schedId] = issued_warp->get_warp_id();
                if (smId == 0)
                    printf("%llu: Selected %u as high prio warp (first)\n", gpu_sim_cycle, issued_warp->get_warp_id());
            }
        }
    }


    if (issued_warp)
    {
        unsigned int tbId = issued_warp->get_cta_id();
        unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
		//if ((smId == 0) && (schedId == 0))
			//printf("%lld: sm 0 sched 0 issued instr\n", gpu_sim_cycle);

        if ((pipeUsed == MEM__OP) && (instrSched->space.get_type() == global_space))
        {
        }
        else
            instrSched->mSetCtaId(0xbeefbeef);

        if (gTBProgressArray)
        {
            assert(activeMaskCount != 0);
            gTBProgressArray[index] += activeMaskCount;
        }

        unsigned int warpId = issued_warp->get_warp_id();
        unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + warpId;

        if (gWarpProgressArray)
            gWarpProgressArray[warpIdx] += activeMaskCount;

		if (gWarpTimeArray)
		{
			if (gWarpTimeArray[warpIdx] == 0)
            	gWarpTimeArray[warpIdx] = gpu_sim_cycle;
		}

        if ((pipeUsed == SP__OP) && rl_scheduler::gTBNumSpInstrsArray)
            rl_scheduler::gTBNumSpInstrsArray[index]++;
        else if ((pipeUsed == SFU__OP) && rl_scheduler::gTBNumSfuInstrsArray)
            rl_scheduler::gTBNumSfuInstrsArray[index]++;
        else if ((pipeUsed == MEM__OP) && rl_scheduler::gTBNumMemInstrsArray)
            rl_scheduler::gTBNumMemInstrsArray[index]++;

        std::string instrStr = ptx_get_insn_str(instrSched->pc);
        if ((pipeUsed == MEM__OP) && (instrSched->space.get_type() == global_space))
        {
            unsigned int index = (issued_warp->get_dynamic_warp_id() << 4) + smId;
            gLastMemInstrMap[index] = instrStr;
        }

        unsigned int issuedWarpTBId = issued_warp->get_cta_id();
        if ((strstr(instrStr.c_str(), "ret;")) || (strstr(instrStr.c_str(), "exit;")))
        {
            if (gNumWarpsAtFinishMapVec.size() != 0)
            {
                std::map<unsigned int, unsigned int>& numWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
                numWarpsAtFinishMap[issuedWarpTBId]++;
            }
        }
        if (strstr(instrStr.c_str(), "bar.sync"))
        {
            if (gNumWarpsAtBarrierMapVec.size() != 0)
            {
                std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
                std::map<unsigned int, unsigned long long>& numCyclesStalledAtBarrierMap = gNumCyclesStalledAtBarrierMapVec.at(smId);
                if (numWarpsAtBarrierMap[issuedWarpTBId] == (gNumWarpsPerBlock - 1))
                {
                    numWarpsAtBarrierMap[issuedWarpTBId] = 0;
            
					// unsigned long long stallCyclesAtBarrier = gpu_sim_cycle - numCyclesStalledAtBarrierMap[issuedWarpTBId];
					// printf("SM %u, TB %u, stall cycles at barrier = %llu\n", smId, issuedWarpTBId, stallCyclesAtBarrier);
					numCyclesStalledAtBarrierMap[issuedWarpTBId] = 0;

					if (gIPAWS)
					{
        				unsigned int tbIdx = smId * MAX_NUM_TB_PER_SM + issuedWarpTBId;
						assert(gWarpsOfTbWaitingAtBarrierSetMap.find(tbIdx) != gWarpsOfTbWaitingAtBarrierSetMap.end());
						std::set<unsigned int>& warpsOfTbWaitingAtBarrierSet = gWarpsOfTbWaitingAtBarrierSetMap[tbIdx];
						for (std::set<unsigned int>::iterator iter = warpsOfTbWaitingAtBarrierSet.begin();
						 	iter != warpsOfTbWaitingAtBarrierSet.end();
						 	iter++)
						{
							unsigned int warpIdx = (*iter);
							gWarpsWaitingAtBarrierSet.erase(warpIdx);
						}
						warpsOfTbWaitingAtBarrierSet.clear();
					}
                }
                else
                {
        			unsigned int warpId = issued_warp->get_warp_id();
        			unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + warpId;

					if (gIPAWS)
					{
        				unsigned int tbIdx = smId * MAX_NUM_TB_PER_SM + issuedWarpTBId;
						if (gWarpsOfTbWaitingAtBarrierSetMap.find(tbIdx) == gWarpsOfTbWaitingAtBarrierSetMap.end())
						{
							std::set<unsigned int> dummySet;
							gWarpsOfTbWaitingAtBarrierSetMap[tbIdx] = dummySet;
						}
						std::set<unsigned int>& warpsOfTbWaitingAtBarrierSet = gWarpsOfTbWaitingAtBarrierSetMap[tbIdx];
						warpsOfTbWaitingAtBarrierSet.insert(warpIdx);
						gWarpsWaitingAtBarrierSet.insert(warpIdx);
					}

                    numWarpsAtBarrierMap[issuedWarpTBId]++;
                    if (numWarpsAtBarrierMap[issuedWarpTBId] == 1)
					{
						assert (numCyclesStalledAtBarrierMap[issuedWarpTBId] == 0);
						numCyclesStalledAtBarrierMap[issuedWarpTBId] = gpu_sim_cycle; //store the cycle at which first warp of this TB reached barrier
					}
                }
            }
        }
    }

}

void scheduler_unit::do_on_warp_issued( unsigned warp_id,
                                        unsigned num_issued,
                                        const std::vector< shd_warp_t* >::const_iterator& prioritized_iter )
{
    m_stats->event_warp_issued( m_shader->get_sid(),
                                warp_id,
                                num_issued,
                                warp(warp_id).get_dynamic_warp_id() );
    warp(warp_id).ibuffer_step();
}

bool scheduler_unit::sort_warps_by_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
            return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_barrier_flag(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			unsigned int lLhsTbId = lhs->get_cta_id();
			unsigned int lRhsTbId = rhs->get_cta_id();

			bool lIsLhsBarrierTb = false;
			if (gBarrierTbIdSet.find(lLhsTbId) != gBarrierTbIdSet.end())
				lIsLhsBarrierTb = true;

			bool lIsRhsBarrierTb = false;
			if (gBarrierTbIdSet.find(lRhsTbId) != gBarrierTbIdSet.end())
				lIsRhsBarrierTb = true;

			if (lIsLhsBarrierTb == lIsRhsBarrierTb)
            	return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id(); //younger warp has more priority
			else if (lIsLhsBarrierTb)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_split_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			bool lLhsSplitWarp = lhs->mIsSplitWarp();
			bool lRhsSplitWarp = rhs->mIsSplitWarp();

			if (lLhsSplitWarp == lRhsSplitWarp)
			{
            	return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id(); //older warp more prio
			}
			else if (lLhsSplitWarp)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_split_youngest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			bool lLhsSplitWarp = lhs->mIsSplitWarp();
			bool lRhsSplitWarp = rhs->mIsSplitWarp();

			if (lLhsSplitWarp == lRhsSplitWarp)
			{
            	return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id(); //younger warp more prio
			}
			else if (lLhsSplitWarp)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_finish_flag(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			unsigned int lLhsTbId = lhs->get_cta_id();
			unsigned int lRhsTbId = rhs->get_cta_id();

			bool lIsLhsFinishTb = false;
			if (gFinishTbIdSet.find(lLhsTbId) != gFinishTbIdSet.end())
				lIsLhsFinishTb = true;

			bool lIsRhsFinishTb = false;
			if (gFinishTbIdSet.find(lRhsTbId) != gFinishTbIdSet.end())
				lIsRhsFinishTb = true;

			if (lIsLhsFinishTb == lIsRhsFinishTb)
            	return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id(); //younger warp has more priority
			else if (lIsLhsFinishTb)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}


bool scheduler_unit::sort_warps_by_youngest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
            return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_yfb_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
            return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
			/**/
			unsigned int lLhsTbId = lhs->get_cta_id();
			unsigned int lRhsTbId = rhs->get_cta_id();

			bool lIsLhsFinishTb = false;
			if (gFinishTbIdSet.find(lLhsTbId) != gFinishTbIdSet.end())
				lIsLhsFinishTb = true;

			bool lIsRhsFinishTb = false;
			if (gFinishTbIdSet.find(lRhsTbId) != gFinishTbIdSet.end())
				lIsRhsFinishTb = true;

			if (lIsLhsFinishTb == lIsRhsFinishTb)
			{
				bool lIsLhsBarrierTb = false;
				if (gBarrierTbIdSet.find(lLhsTbId) != gBarrierTbIdSet.end())
					lIsLhsBarrierTb = true;

				bool lIsRhsBarrierTb = false;
				if (gBarrierTbIdSet.find(lRhsTbId) != gBarrierTbIdSet.end())
					lIsRhsBarrierTb = true;

				if (lIsLhsBarrierTb == lIsRhsBarrierTb)
            		return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
				else if (lIsLhsBarrierTb)
					return true;
				else
					return false;
			}
			else if (lIsLhsFinishTb)
				return true;
			else
				return false;
		/**/
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_long_lat_mem_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			bool lLhsLongLatMemInstr = lhs->mIsLongLatMemInstr();
			bool lRhsLongLatMemInstr = rhs->mIsLongLatMemInstr();

			if (lLhsLongLatMemInstr == lRhsLongLatMemInstr)
			{
            	return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id(); //older warp more prio
			}
			else if (lLhsLongLatMemInstr)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_long_lat_mem_youngest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			bool lLhsLongLatMemInstr = lhs->mIsLongLatMemInstr();
			bool lRhsLongLatMemInstr = rhs->mIsLongLatMemInstr();

			if (lLhsLongLatMemInstr == lRhsLongLatMemInstr)
			{
            	return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id(); //younger warp more prio
			}
			else if (lLhsLongLatMemInstr)
				return true;
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

operation_pipeline_t rl_scheduler::mGetCmdPipeType(shd_warp_t* xWarp)
{
	operation_pipeline_t cmdPipeType = UNKNOWN_OP;
    if ((xWarp != NULL) && (xWarp->waiting() == false) && (xWarp->ibuffer_empty() == false) && (xWarp->done_exit() == false))
    {
        const warp_inst_t* pI = xWarp->ibuffer_next_inst();
        if(pI) 
        {
            unsigned pc, rpc;
            unsigned int warp_id = xWarp->get_warp_id();
            rl_scheduler::gSimtStack[warp_id]->get_pdom_stack_top_info(&pc, &rpc);

            if( pc == pI->pc ) 
            {
                if (!rl_scheduler::gScoreboard->checkCollision(warp_id, pI)) 
                {
                    if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP)) 
						cmdPipeType = MEM__OP;
					else
					{
                        bool sp_pipe_avail = m_sp_out->has_free();
                        bool sfu_pipe_avail = m_sfu_out->has_free();
                        if (sp_pipe_avail && (pI->op != SFU_OP)) 
							cmdPipeType = SP__OP;
                        else if (sfu_pipe_avail && ((pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)))
							cmdPipeType = SFU__OP;
					}
                }
            }
        }
    }
	return cmdPipeType;
}

bool scheduler_unit::sort_warps_by_mfs_cmd_type(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			operation_pipeline_t lhsCmdPipeType = rl_scheduler::gCurrRLSchedulerUnit->mGetCmdPipeType(lhs);
			operation_pipeline_t rhsCmdPipeType = rl_scheduler::gCurrRLSchedulerUnit->mGetCmdPipeType(rhs);
			switch (lhsCmdPipeType)
			{
				case MEM__OP:
					if (rhsCmdPipeType == MEM__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else
						return true;
					break;
				case SFU__OP:
					if (rhsCmdPipeType == MEM__OP)
						return false;
					else if (rhsCmdPipeType == SFU__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else
						return true;
					break;
				case SP__OP:
					if ((rhsCmdPipeType == MEM__OP) || (rhsCmdPipeType == SFU__OP))
						return false;
					else if (rhsCmdPipeType == SP__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else
						return true;
					break;
				case UNKNOWN_OP:
					if (rhsCmdPipeType != UNKNOWN_OP)
						return false;
					else
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					break;
			}
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_fms_cmd_type(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			operation_pipeline_t lhsCmdPipeType = rl_scheduler::gCurrRLSchedulerUnit->mGetCmdPipeType(lhs);
			operation_pipeline_t rhsCmdPipeType = rl_scheduler::gCurrRLSchedulerUnit->mGetCmdPipeType(rhs);
			switch (lhsCmdPipeType)
			{
				case MEM__OP:
					if (rhsCmdPipeType == MEM__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else if (rhsCmdPipeType == SFU__OP)
						return false;
					else
						return true;
					break;
				case SFU__OP:
					if (rhsCmdPipeType == SFU__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else 
						return true;
					break;
				case SP__OP:
					if ((rhsCmdPipeType == MEM__OP) || (rhsCmdPipeType == SFU__OP))
						return false;
					else if (rhsCmdPipeType == SP__OP)
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					else
						return true;
					break;
				case UNKNOWN_OP:
					if (rhsCmdPipeType != UNKNOWN_OP)
						return false;
					else
						return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
					break;
			}
			if (lhsCmdPipeType == rhsCmdPipeType)
            	return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
			else if (lhsCmdPipeType == SFU__OP)
				return true;
			else if (lhsCmdPipeType == MEM__OP)
			{
				if (rhsCmdPipeType == SFU__OP)
					return false;
				else
					return true;
			}
			else
				return false;
        }
    } else {
        return lhs < rhs;
    }
}

bool scheduler_unit::sort_warps_by_iter_num(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) {
        if ( lhs->done_exit() || lhs->waiting() ) {
            return false;
        } else if ( rhs->done_exit() || rhs->waiting() ) {
            return true;
        } else {
			unsigned lhsIterNum = lhs->mGetCurrIterNum();
			unsigned rhsIterNum = rhs->mGetCurrIterNum();
			if (lhsIterNum == rhsIterNum)
			{
				//return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
				unsigned int lhsCurrIterRank = lhs->mGetCurrIterRank();
				unsigned int rhsCurrIterRank = rhs->mGetCurrIterRank();
				return lhsCurrIterRank > rhsCurrIterRank;
			}
			else return lhsIterNum < rhsIterNum;
        }
    } else {
        return lhs < rhs;
    }
}

unsigned int scheduler_unit::getCmdPipeType(const warp_inst_t* pI)
{
    unsigned int actionVal = SCHED_NO_INSTR;
	assert(pI);

    if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP))
	{
		if (m_mem_out->has_free())
		{
        	if (pI->space.get_type() == global_space)
				actionVal = SCHED_GMEM_INSTR;
			else
				actionVal = SCHED_STC_MEM_INSTR;
		}
	}
	else
	{
        bool sfu_pipe_avail = m_sfu_out->has_free();
		bool sp_pipe_avail = m_sp_out->has_free();

    	if (sp_pipe_avail && (pI->op != SFU_OP))
        	actionVal = SCHED_SP_INSTR;
    	else if (sfu_pipe_avail && ((pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)))
        	actionVal = SCHED_SFU_INSTR;
	}
    return actionVal;
}

unsigned int rl_scheduler::mGetActionValue(shd_warp_t* xWarp, unsigned int xActionType)
{
	unsigned int lActionValue = 0xdeaddead;
    const warp_inst_t* pI;
    if ((xWarp != NULL) && (xWarp->waiting() == false) && (xWarp->ibuffer_empty() == false) && (xWarp->done_exit() == false))
    {
        pI = xWarp->ibuffer_next_inst();
        if(pI) 
        {
            unsigned pc, rpc;
            unsigned int warp_id = xWarp->get_warp_id();
            rl_scheduler::gSimtStack[warp_id]->get_pdom_stack_top_info(&pc, &rpc);

            if( pc == pI->pc ) 
            {
                if (!rl_scheduler::gScoreboard->checkCollision(warp_id, pI)) 
                {
					bool skip = true;
                    if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP)) 
                    {
                        if (m_mem_out->has_free())
							skip = false;
					}
					else
					{
                        bool sp_pipe_avail = m_sp_out->has_free();
                        bool sfu_pipe_avail = m_sfu_out->has_free();
                        if (sp_pipe_avail && (pI->op != SFU_OP)) 
							skip = false;
                        else if (sfu_pipe_avail && ((pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)))
							skip = false;
					}
					if (skip == false)
					{
						if (xActionType == USE_NAM_ACTION)
						{
    						lActionValue = SCHED_NO_INSTR;
							assert(pI);

    						if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP))
							{
								if (m_mem_out->has_free())
									lActionValue = SCHED_MEM_INSTR;
							}
							else
							{
        						bool sfu_pipe_avail = m_sfu_out->has_free();
								bool sp_pipe_avail = m_sp_out->has_free();
						
    							if (sp_pipe_avail && (pI->op != SFU_OP))
        							lActionValue = SCHED_ALU_INSTR;
    							else if (sfu_pipe_avail && ((pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)))
        							lActionValue = SCHED_ALU_INSTR;
							}
						}
                    	else if (xActionType == USE_TB_CMD_PIPE_AS_ACTION)
                    	{
                        	unsigned int tbId = xWarp->get_cta_id();
                        	unsigned int pipeUsed = rl_scheduler::gCurrRLSchedulerUnit->getCmdPipeType(pI);
							lActionValue = tbId * MAX_ACTIONS_OF_TYPE_CMD_PIPE + pipeUsed;
                    	}
                    	else if (xActionType == USE_TB_ID_AS_ACTION)
                        	lActionValue = xWarp->get_cta_id();
                    	else if (xActionType == USE_WARP_ID_AS_ACTION)
                    	{
                        	lActionValue = xWarp->get_warp_id();
                        	if ((lActionValue & 0x1) == 0)
                            	lActionValue = (lActionValue >> 1);
                        	else
                            	lActionValue = ((lActionValue - 1) >> 1);
                    	}
                    	else if (xActionType == USE_TB_WARP_ID_AS_ACTION)
                    	{
                        	unsigned int tbId = xWarp->get_cta_id();
                        	unsigned int warpId = xWarp->get_warp_id();
                        	if ((warpId & 0x1) == 0)
                            	warpId = (warpId >> 1);
                        	else
                            	warpId = ((warpId - 1) >> 1);
                        	//lActionValue = (tbId << 2) | warpId;
							lActionValue = tbId * MAX_ACTIONS_OF_TYPE_WARP_ID + warpId;
                    	}
						else if (xActionType == USE_TB_TYPE_AS_ACTION)
						{
							lActionValue = SCHED_NO_TB;
    						unsigned int smId = rl_scheduler::gCurrRLSchedulerUnit->get_sid();
        					std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
							for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
				 				iter != lNumWarpsAtFinishMap.end();
				 				iter++)
							{
								unsigned int lNumWarpsAtFinish = iter->second;
								if (lNumWarpsAtFinish > 0)
								{
									unsigned int lTbId = iter->first;
									if (lTbId == xWarp->get_cta_id())
									{
										lActionValue = SCHED_FINISH_TB;
										break;
									}
								}
							}
							if (lActionValue == SCHED_NO_TB)
							{
        						std::map<unsigned int, unsigned int>& lNumWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
								for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtBarrierMap.begin();
				 					iter != lNumWarpsAtBarrierMap.end();
				 					iter++)
								{
									unsigned int lNumWarpsAtBarrier = iter->second;
									if (lNumWarpsAtBarrier > 0)
									{
										unsigned int lTbId = iter->first;
										if (lTbId == xWarp->get_cta_id())
										{
											lActionValue = SCHED_BARRIER_TB;
											break;
										}
									}
								}
							}
							if (lActionValue == SCHED_NO_TB)
							{
								unsigned int lSchedTbId = xWarp->get_cta_id();
                				unsigned index = smId * MAX_NUM_TB_PER_SM + lSchedTbId;
								unsigned int lSchedTbProg = gTBProgressArray[index];
								lActionValue = SCHED_FASTEST_TB;
	
            					for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            					{
                					unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                					unsigned int tbProgress = gTBProgressArray[index];
					
                					if ((tbProgress > lSchedTbProg) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                					{
                    					lActionValue = SCHED_NO_TB;
										break;
                					}
            					}
							}
							if (lActionValue == SCHED_NO_TB)
							{
								unsigned int lSchedTbId = xWarp->get_cta_id();
                				unsigned index = smId * MAX_NUM_TB_PER_SM + lSchedTbId;
								unsigned int lSchedTbProg = gTBProgressArray[index];
								lActionValue = SCHED_SLOWEST_TB;
	
            					for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            					{
                					unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                					unsigned int tbProgress = gTBProgressArray[index];
					
                					if ((tbProgress < lSchedTbProg) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                					{
                    					lActionValue = SCHED_NO_TB;
										break;
                					}
            					}
							}
						}
						else if (xActionType == USE_NUM_WARPS_AS_ACTION)
						{
							assert(0);
						}
						else if (xActionType == USE_L1_BYPASS_AS_ACTION)
						{
							assert(0);
						}
						else if (xActionType == USE_WHICH_SCHED_AS_ACTION)
						{
							assert(0);
						}
						else if (xActionType == USE_WHICH_WARP_AS_ACTION)
						{
							assert(0);
						}
						else if (xActionType == USE_WHICH_WARP_TYPE_AS_ACTION)
						{
							assert(0);
						}
						else if (xActionType == USE_LRR_GTO_AS_ACTION)
						{
							assert(0);
						}
                    	else
                    	{
                        	assert(xActionType == USE_CMD_PIPE_AS_ACTION);
                        	lActionValue = rl_scheduler::gCurrRLSchedulerUnit->getCmdPipeType(pI);
                    	}
					}
                }
            }
        }
    }
	return lActionValue;
}

bool scheduler_unit::sort_warps_by_highest_q_value(shd_warp_t* lhs, shd_warp_t* rhs)
{
    if (rhs && lhs) 
    {
		unsigned int lLhsDynWarpId = lhs->get_dynamic_warp_id();
		unsigned int lRhsDynWarpId = rhs->get_dynamic_warp_id();

        if ((lhs->done_exit() || lhs->waiting()) && (rhs->done_exit() || rhs->waiting()))
            return (lLhsDynWarpId < lRhsDynWarpId);

        if (lhs->done_exit() || lhs->waiting())
            return false;

        if (rhs->done_exit() || rhs->waiting()) 
            return true;

        unsigned int lhsActionVal = rl_scheduler::gCurrRLSchedulerUnit->mGetActionValue(lhs, rl_scheduler::dRLActionTypes[0]);
        unsigned int rhsActionVal = rl_scheduler::gCurrRLSchedulerUnit->mGetActionValue(rhs, rl_scheduler::dRLActionTypes[0]);

        if ((lhsActionVal == 0xdeaddead) && (rhsActionVal == 0xdeaddead))
            return (lLhsDynWarpId < lRhsDynWarpId);

        if (lhsActionVal == 0xdeaddead)
            return false;

        if (rhsActionVal == 0xdeaddead)
            return true;

		float lhsQ = 0.0;
		float rhsQ = 0.0;
        unsigned long long index = 0xdeaddead;
		if (lhsActionVal != rhsActionVal)
		{
        	index = (rl_scheduler::gPrimaryStateVal * rl_scheduler::gPrimaryNumActions) + lhsActionVal;
        	lhsQ =  gPrimaryRLEngine->mGetQvalue(index);

        	index = (rl_scheduler::gPrimaryStateVal * rl_scheduler::gPrimaryNumActions) + rhsActionVal;
        	rhsQ =  gPrimaryRLEngine->mGetQvalue(index);
		}

		double EPS = 1.0e-4;
		if (fabs(lhsQ - rhsQ) < EPS)
		{
			//if (rl_scheduler::gSecondaryQvalues != 0)
			if (gSecondaryRLEngine != 0)
			{
        		lhsActionVal = rl_scheduler::gCurrRLSchedulerUnit->mGetActionValue(lhs, rl_scheduler::dRLActionTypes[1]);
        		rhsActionVal = rl_scheduler::gCurrRLSchedulerUnit->mGetActionValue(rhs, rl_scheduler::dRLActionTypes[1]);
	
        		if ((lhsActionVal == 0xdeaddead) && (rhsActionVal == 0xdeaddead))
            		return (lLhsDynWarpId < lRhsDynWarpId);
	
        		if (lhsActionVal == 0xdeaddead)
            		return false;
	
        		if (rhsActionVal == 0xdeaddead)
            		return true;
	
				if (lhsActionVal != rhsActionVal)
				{
        			index = (rl_scheduler::gSecondaryStateVal * rl_scheduler::gSecondaryNumActions) + lhsActionVal;
        			lhsQ =  gSecondaryRLEngine->mGetQvalue(index);
	
        			index = (rl_scheduler::gSecondaryStateVal * rl_scheduler::gSecondaryNumActions) + rhsActionVal;
        			rhsQ =  gSecondaryRLEngine->mGetQvalue(index);
				}
			}
		}
        return (lhsQ > (rhsQ + EPS));
    } 
    else 
    {
        return lhs < rhs;
    }
}

void random_scheduler::order_warps()
{
    m_next_cycle_prioritized_warps.clear();

    std::vector<shd_warp_t*> warpVec = m_supervised_warps;

    if (warpVec.size() > 0)
    {
        long int randVal = random();
        unsigned int idx = randVal % warpVec.size();
        shd_warp_t* warp = warpVec[idx];
        m_next_cycle_prioritized_warps.push_back(warp);
    
        for (unsigned i = (idx + 1); i < warpVec.size(); ++i)
            m_next_cycle_prioritized_warps.push_back(warpVec[i]);
    
        for (unsigned i = 0; i < idx; ++i)
            m_next_cycle_prioritized_warps.push_back(warpVec[i]);
    }
}

void lrr_scheduler::order_warps()
{
    order_lrr( m_next_cycle_prioritized_warps,
               m_supervised_warps,
               m_last_supervised_issued,
               m_supervised_warps.size() );
}

void gto_scheduler::order_warps()
{
    order_by_priority( m_next_cycle_prioritized_warps,
                       m_supervised_warps,
                       m_last_supervised_issued,
                       m_supervised_warps.size(),
                       ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       scheduler_unit::sort_warps_by_oldest_dynamic_id );
}

void ipaws_scheduler::order_warps()
{
	if (gTotalNumWarpsFinished == 0)
	{
		//adapt phase
		for (std::set<unsigned int>::iterator iter = gWarpsWaitingAtBarrierSet.begin();
		     iter != gWarpsWaitingAtBarrierSet.end();
			 iter++)
		{
			unsigned int warpIdx = *iter;
			gWarpBarrierTimeArray[warpIdx]++;
		}
    	order_by_priority( m_next_cycle_prioritized_warps,
                       	m_supervised_warps,
                       	m_last_supervised_issued,
                       	m_supervised_warps.size(),
                       	ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       	scheduler_unit::sort_warps_by_oldest_dynamic_id );
	}
	else
	{
		if (gIPAWS_RecoverPhase)
		{
    		order_by_priority(m_next_cycle_prioritized_warps,
                     		  m_supervised_warps,
                      		  m_last_supervised_issued,
                      		  m_supervised_warps.size(),
                      		  ORDERED_PRIORITY_FUNC_ONLY,
                       		  scheduler_unit::sort_warps_by_youngest_dynamic_id);
		}
		else if (gIPAWS_UseGTO)
		{
    		order_by_priority( m_next_cycle_prioritized_warps,
                       		m_supervised_warps,
                       		m_last_supervised_issued,
                       		m_supervised_warps.size(),
                       		ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       		scheduler_unit::sort_warps_by_oldest_dynamic_id );
		}
		else //
		{
    		order_lrr( m_next_cycle_prioritized_warps,
               		m_supervised_warps,
               		m_last_supervised_issued,
               		m_supervised_warps.size() );
		}
	}
}

void rto_scheduler::order_warps()
{
    order_by_priority( m_next_cycle_prioritized_warps,
                       m_supervised_warps,
                       m_last_supervised_issued,
                       m_supervised_warps.size(),
                       ORDERING_RANDOM_GREEDY_THEN_PRIORITY_FUNC,
                       scheduler_unit::sort_warps_by_oldest_dynamic_id );
}



void
two_level_active_scheduler::do_on_warp_issued( unsigned warp_id,
                                               unsigned num_issued,
                                               const std::vector< shd_warp_t* >::const_iterator& prioritized_iter )
{
    scheduler_unit::do_on_warp_issued( warp_id, num_issued, prioritized_iter );
    if ( SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization ) {
        std::vector< shd_warp_t* > new_active; 
        order_lrr( new_active,
                   m_next_cycle_prioritized_warps,
                   prioritized_iter,
                   m_next_cycle_prioritized_warps.size() );
        m_next_cycle_prioritized_warps = new_active;
    } else {
        fprintf( stderr,
                 "Unimplemented m_inner_level_prioritization: %d\n",
                 m_inner_level_prioritization );
        abort();
    }
}

void two_level_active_scheduler::order_warps()
{
    //Move waiting warps to m_pending_warps
    unsigned num_demoted = 0;
    for (   std::vector< shd_warp_t* >::iterator iter = m_next_cycle_prioritized_warps.begin();
            iter != m_next_cycle_prioritized_warps.end(); ) {
        bool waiting = (*iter)->waiting();
        for (int i=0; i<4; i++){
            const warp_inst_t* inst = (*iter)->ibuffer_next_inst();
            //Is the instruction waiting on a long operation?
            if ( inst && inst->in[i] > 0 && this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in[i])){
                waiting = true;
            }
        }

        if( waiting ) {
            m_pending_warps.push_back(*iter);
            iter = m_next_cycle_prioritized_warps.erase(iter);
            SCHED_DPRINTF( "DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                           (*iter)->get_warp_id(),
                           (*iter)->get_dynamic_warp_id() );
            ++num_demoted;
        } else {
            ++iter;
        }
    }

    //If there is space in m_next_cycle_prioritized_warps, promote the next m_pending_warps
    unsigned num_promoted = 0;
    if ( SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization ) {
        while ( m_next_cycle_prioritized_warps.size() < m_max_active_warps ) {
            m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
            m_pending_warps.pop_front();
            SCHED_DPRINTF( "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
                           (m_next_cycle_prioritized_warps.back())->get_warp_id(),
                           (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id() );
            ++num_promoted;
        }
    } else {
        fprintf( stderr,
                 "Unimplemented m_outer_level_prioritization: %d\n",
                 m_outer_level_prioritization );
        abort();
    }
    assert( num_promoted == num_demoted );
}

swl_scheduler::swl_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                               Scoreboard* scoreboard, simt_stack** simt,
                               std::vector<shd_warp_t>* warp,
                               register_set* sp_out,
                               register_set* sfu_out,
                               register_set* mem_out,
                               int id,
                               char* config_string )
    : scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id )
{
    unsigned m_prioritization_readin;
    int ret = sscanf( config_string,
                      "warp_limiting:%d:%d",
                      &m_prioritization_readin,
                      &m_num_warps_to_limit
                     );
    assert( 2 == ret );
    m_prioritization = (scheduler_prioritization_type)m_prioritization_readin;
    // Currently only GTO is implemented
    assert( m_prioritization == SCHEDULER_PRIORITIZATION_GTO );
    assert( m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader );
}

void swl_scheduler::order_warps()
{
    if ( SCHEDULER_PRIORITIZATION_GTO == m_prioritization ) {
        order_by_priority( m_next_cycle_prioritized_warps,
                           m_supervised_warps,
                           m_last_supervised_issued,
                           MIN( m_num_warps_to_limit, m_supervised_warps.size() ),
                           ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                           scheduler_unit::sort_warps_by_oldest_dynamic_id );
    } else {
        fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
        abort();
    }
}

void rlEngine::computeReward(bool instIssued)
{
	static unsigned long long prev_gpu_sim_cycle = 1;
    dReward = instIssued ? gReward : gPenalty;
	gTotalGPUWideReward += dReward;
	gTotalGPUWideDiscountedReward += (dDiscountFactor * dReward);
	if (prev_gpu_sim_cycle == (gpu_sim_cycle - 1))
	{
		dDiscountFactor *= gGamma;
		prev_gpu_sim_cycle = gpu_sim_cycle;
	}
}

unsigned int rlEngine::computeAction(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
    unsigned int action = 0xdeaddead;

    if (dActionType == USE_TB_CMD_PIPE_AS_ACTION)
    {
        unsigned int tbId = gNumTBsPerSM;
        if (warp)
            tbId = warp->get_cta_id();
		action = tbId * MAX_ACTIONS_OF_TYPE_CMD_PIPE + pipeUsed;
    }
    else if (dActionType == USE_TB_ID_AS_ACTION)
    {
        if (warp)
            action = warp->get_cta_id();
        else
            action = gNumTBsPerSM;
    }
    else if (dActionType == USE_WARP_ID_AS_ACTION)
    {
        action = DUMMY_WARP_ID;
        if (warp)
            action = warp->get_warp_id();

        //warpId is the action and for action 5 bits are being used as each scheduler has only 24 warps
        //so do the following magic to make the warpId between 0-23
        if ((action & 0x1) == 0)
            action = (action >> 1);
        else
            action = ((action - 1) >> 1);
    }
    else if (dActionType == USE_TB_WARP_ID_AS_ACTION)
    {
        unsigned int tbId = gNumTBsPerSM;
        unsigned int warpId = DUMMY_WARP_ID;
        if (warp)
        {
            tbId = warp->get_cta_id();
            warpId = warp->get_warp_id();
        }
        if ((warpId & 0x1) == 0)
            warpId = (warpId >> 1);
        else
            warpId = ((warpId - 1) >> 1);
		action = tbId * MAX_ACTIONS_OF_TYPE_WARP_ID + warpId;
    }
    else if (dActionType == USE_TB_TYPE_AS_ACTION)
	{
		action = SCHED_NO_TB;
		if (warp)
		{
			//find out which tb type this warp belongs to
			//is it a warp of a tb which has 'finished' warps
			//is it a warp of a tb which has warps at barrier
			//is it a warp of the fastest tb
			//is it a warp of the slowest tb

    		unsigned int smId = dRLSched->m_shader->get_sid();
        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
			 	iter != lNumWarpsAtFinishMap.end();
			 	iter++)
			{
				unsigned int lNumWarpsAtFinish = iter->second;
				if (lNumWarpsAtFinish > 0)
				{
					unsigned int lTbId = iter->first;
					if (lTbId == warp->get_cta_id())
					{
						action = SCHED_FINISH_TB;
						break;
					}
				}
			}
			if (action == SCHED_NO_TB)
			{
        		std::map<unsigned int, unsigned int>& lNumWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
				for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtBarrierMap.begin();
				 	iter != lNumWarpsAtBarrierMap.end();
				 	iter++)
				{
					unsigned int lNumWarpsAtBarrier = iter->second;
					if (lNumWarpsAtBarrier > 0)
					{
						unsigned int lTbId = iter->first;
						if (lTbId == warp->get_cta_id())
						{
							action = SCHED_BARRIER_TB;
							break;
						}
					}
				}
			}
			if (action == SCHED_NO_TB)
			{
				unsigned int lSchedTbId = warp->get_cta_id();
         		unsigned index = smId * MAX_NUM_TB_PER_SM + lSchedTbId;
				unsigned int lSchedTbProg = gTBProgressArray[index];
				action = SCHED_FASTEST_TB;
	
            	for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            	{
                	unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                	unsigned int tbProgress = gTBProgressArray[index];
					
                	if ((tbProgress > lSchedTbProg) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                	{
                    	action = SCHED_NO_TB;
						break;
                	}
            	}
			}
			if (action == SCHED_NO_TB)
			{
				unsigned int lSchedTbId = warp->get_cta_id();
            	unsigned index = smId * MAX_NUM_TB_PER_SM + lSchedTbId;
				unsigned int lSchedTbProg = gTBProgressArray[index];
				action = SCHED_SLOWEST_TB;
	
				for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            	{
                	unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                	unsigned int tbProgress = gTBProgressArray[index];
					
                	if ((tbProgress < lSchedTbProg) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                	{
                    	action = SCHED_NO_TB;
						break;
                	}
        		}
			}
		}
	}
    else if (dActionType == USE_CMD_PIPE_AS_ACTION)
    {
		if (instrSched == 0)
			action = SCHED_NO_INSTR;
        else
		{
    		action = SCHED_NO_INSTR;

    		if (pipeUsed == MEM__OP)
			{
        		if (instrSched->space.get_type() == global_space)
					action = SCHED_GMEM_INSTR;
				else
					action = SCHED_STC_MEM_INSTR;
			}
			else
			{
    			if (pipeUsed ==  SP__OP)
        			action = SCHED_SP_INSTR;
    			else if (pipeUsed == SFU__OP)
        			action = SCHED_SFU_INSTR;
			}
		}
    }
	else if (dActionType == USE_NUM_WARPS_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_L1_BYPASS_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_WHICH_SCHED_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_WHICH_WARP_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_WHICH_WARP_TYPE_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_LRR_GTO_AS_ACTION)
	{
		action = dCurrAction;
	}
	else if (dActionType == USE_NAM_ACTION)
	{
		action = dCurrAction;
	}
    else
    	assert(0);
    return action;
}

void rl_scheduler::collectRewardAndSetAttributes(bool instIssued, shd_warp_t* warp, 
                                               operation_pipeline_t pipeUsed, char* instrType,
                                               warp_inst_t* instrSched)
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->collectRewardAndSetAttributes(instIssued, warp, pipeUsed, instrType, instrSched);
}

void rlEngine::collectRewardAndSetAttributes(bool instIssued, shd_warp_t* warp, 
                                             operation_pipeline_t pipeUsed, char* instrType,
                                             warp_inst_t* instrSched)
{
    computeReward(instIssued);

    unsigned int action = computeAction(warp, pipeUsed, instrSched);

	dCurrState = getCurrStateForAction(action);

    dPrevState = dCurrState;
    dPrevAction = action;

    setAttributeValues1(warp);
}

void rl_scheduler::initCurrStateAndAction()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->initCurrStateAndAction();
}

void rlEngine::initCurrStateAndAction()
{
	unsigned long long lCurrState = 0;
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];

        unsigned int defVal = attr.defaultValue;
		dCurrStateVector[i] = defVal;
        lCurrState += (defVal * attr.dPlaceValue);
    }
	dCurrState = lCurrState;
}

#define HIGH_NUM_WARPS_EXECUTING_MEM_INSTR 450

unsigned int rl_attribute::mGetBucketValue(unsigned int lNumBucketsToBeUsed)
{
	int lMaxValue = this->bucketSize * this->numAttrValues;

	unsigned int lRetVal;
	if (gUniformBuckets)
	{
		//int lRawValue = this->currRawValue;
		int lRand = (random() % 3) - 1;
		//int lRawValue = this->currRawValue + (lRand * ((10 * lMaxValue) / 100));
		int offset = (10 * this->bucketSize) / 100;
		if (offset == 0)
			offset = 1;
		int lRawValue = this->currRawValue + (lRand * offset);

		if (lNumBucketsToBeUsed == 8)
		{
			if (lRawValue < (lMaxValue * 12 / 100))
	       		lRetVal = 0;
			else if (lRawValue < (lMaxValue * 25 / 100))
	       		lRetVal = 1;
			else if (lRawValue < (lMaxValue * 37 / 100))
	       		lRetVal = 2;
			else if (lRawValue < (lMaxValue * 50 / 100))
	       		lRetVal = 3;
			else if (lRawValue < (lMaxValue * 62 / 100))
	       		lRetVal = 4;
			else if (lRawValue < (lMaxValue * 75 / 100))
	       		lRetVal = 5;
			else if (lRawValue < (lMaxValue * 87 / 100))
	       		lRetVal = 6;
			else
	       		lRetVal = 7;
		}
		else if (lNumBucketsToBeUsed == 6)
		{
			if (lRawValue < (lMaxValue * 16 / 100))
	       		lRetVal = 0;
			else if (lRawValue < (lMaxValue * 33 / 100))
	       		lRetVal = 1;
			else if (lRawValue < (lMaxValue * 49 / 100))
	       		lRetVal = 2;
			else if (lRawValue < (lMaxValue * 66 / 100))
	       		lRetVal = 3;
			else if (lRawValue < (lMaxValue * 83 / 100))
	       		lRetVal = 4;
			else
	       		lRetVal = 5;
		}
		else if (lNumBucketsToBeUsed == 5)
		{
			if (lRawValue < (lMaxValue * 20 / 100))
	       		lRetVal = 0;
			else if (lRawValue < (lMaxValue * 40 / 100))
	       		lRetVal = 1;
			else if (lRawValue < (lMaxValue * 60 / 100))
	       		lRetVal = 2;
			else if (lRawValue < (lMaxValue * 80 / 100))
	       		lRetVal = 3;
			else
	       		lRetVal = 4;
		}
		else if (lNumBucketsToBeUsed == 4)
		{
			if (lRawValue < (lMaxValue * 25 / 100))
	       		lRetVal = 0;
			else if (lRawValue < (lMaxValue * 50 / 100))
	       		lRetVal = 1;
			else if (lRawValue < (lMaxValue * 75 / 100))
	       		lRetVal = 2;
			else
	       		lRetVal = 3;
		}
		else if (lNumBucketsToBeUsed == 3)
		{
			if (lRawValue < (lMaxValue * 33 / 100))
	       		lRetVal = 0;
			else if (lRawValue < (lMaxValue * 67 / 100))
	       		lRetVal = 1;
			else
	       		lRetVal = 2;
		}
		else if (lNumBucketsToBeUsed == 2)
		{
			if (lRawValue < (lMaxValue * 50 / 100))
	       		lRetVal = 0;
			else
	       		lRetVal = 1;
		}
		else 
			assert(0);
	}
	else
	{
		int lRand = (random() % 3) - 1;
		int lRawValue = this->currRawValue + (lRand * ((10 * lMaxValue) / 100));

		if (dDecreasingBucketSizes == 1)
		{
			if (lNumBucketsToBeUsed == 8)
			{
				if (lRawValue < (lMaxValue * 20 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 38 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 53 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 66 / 100))
	       			lRetVal = 3;
				else if (lRawValue < (lMaxValue * 78 / 100))
	       			lRetVal = 4;
				else if (lRawValue < (lMaxValue * 88 / 100))
	       			lRetVal = 5;
				else if (lRawValue < (lMaxValue * 95 / 100))
	       			lRetVal = 6;
				else
	       			lRetVal = 7;
			}
			else if (lNumBucketsToBeUsed == 6)
			{
				if (lRawValue < (lMaxValue * 27 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 50 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 70 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 85 / 100))
	       			lRetVal = 3;
				else if (lRawValue < (lMaxValue * 95 / 100))
	       			lRetVal = 4;
				else
	       			lRetVal = 5;
			}
			else if (lNumBucketsToBeUsed == 5)
			{
				if (lRawValue < (lMaxValue * 30 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 55 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 75 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 90 / 100))
	       			lRetVal = 3;
				else
	       			lRetVal = 4;
			}
			else if (lNumBucketsToBeUsed == 4)
			{
				if (lRawValue < (lMaxValue * 40 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 70 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 90 / 100))
	       			lRetVal = 2;
				else
	       			lRetVal = 3;
			}
			else if (lNumBucketsToBeUsed == 3)
			{
				if (lRawValue < (lMaxValue * 50 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 80 / 100))
	       			lRetVal = 1;
				else
	       			lRetVal = 2;
			}
			else if (lNumBucketsToBeUsed == 2)
			{
				if (lRawValue < (lMaxValue * 60 / 100))
	       			lRetVal = 0;
				else
	       			lRetVal = 1;
			}
			else 
				assert(0);
		}
		else
		{
			if (lNumBucketsToBeUsed == 8)
			{
				if (lRawValue < (lMaxValue * 5 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 12 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 22 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 34 / 100))
	       			lRetVal = 3;
				else if (lRawValue < (lMaxValue * 47 / 100))
	       			lRetVal = 4;
				else if (lRawValue < (lMaxValue * 62 / 100))
	       			lRetVal = 5;
				else if (lRawValue < (lMaxValue * 80 / 100))
	       			lRetVal = 6;
				else
	       			lRetVal = 7;
			}
			else if (lNumBucketsToBeUsed == 6)
			{
				if (lRawValue < (lMaxValue * 5 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 15 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 30 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 50 / 100))
	       			lRetVal = 3;
				else if (lRawValue < (lMaxValue * 73 / 100))
	       			lRetVal = 4;
				else
	       			lRetVal = 5;
			}
			else if (lNumBucketsToBeUsed == 5)
			{
				if (lRawValue < (lMaxValue * 10 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 25 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 45 / 100))
	       			lRetVal = 2;
				else if (lRawValue < (lMaxValue * 70 / 100))
	       			lRetVal = 3;
				else
	       			lRetVal = 4;
			}
			else if (lNumBucketsToBeUsed == 4)
			{
				if (lRawValue < (lMaxValue * 10 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 30 / 100))
	       			lRetVal = 1;
				else if (lRawValue < (lMaxValue * 60 / 100))
	       			lRetVal = 2;
				else
	       			lRetVal = 3;
			}
			else if (lNumBucketsToBeUsed == 3)
			{
				if (lRawValue < (lMaxValue * 20 / 100))
	       			lRetVal = 0;
				else if (lRawValue < (lMaxValue * 50 / 100))
	       			lRetVal = 1;
				else
	       			lRetVal = 2;
			}
			else if (lNumBucketsToBeUsed == 2)
			{
				if (lRawValue < (lMaxValue * 40 / 100))
	       			lRetVal = 0;
				else
	       			lRetVal = 1;
			}
			else 
				assert(0);
		}
	}
	return lRetVal;
}

void rl_attribute::mSetBucketizedAttrValue(unsigned int xRawValue)
{
	unsigned int lMaxValue = this->bucketSize * this->numAttrValues;
	if (xRawValue > lMaxValue)
		xRawValue = lMaxValue;
	// if (xRawValue > 100)
		// xRawValue = 100;
	this->currRawValue = xRawValue;
	if (rl_scheduler::gUseCMACFuncApprox == false)
	{
		assert(dDecreasingBucketSizes != 0xdeaddead);
		this->currValue = mGetBucketValue(this->numAttrValues);
	}
}

void rl_scheduler::setAttributeValues2()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->setAttributeValues2();
}

void rlEngine::setAttributeValues2()
{
    unsigned int smId = dRLSched->m_shader->get_sid();
    assert(smId < gNumSMs);
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];
        if ((attr.attrType == LAST_WARP_ISSUED) ||
            (attr.attrType == LAST_TB_ISSUED))
        {
            //this will be set in setAttributeValues1
        }
		else if (attr.attrType == FAST_TB)
		{
            unsigned int maxProgress = 0;
            unsigned int maxProgressTB = 0xdeaddead;
            attr.currValue = gNumTBsPerSM;

            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                unsigned int tbProgress = gTBProgressArray[index];

                if ((tbProgress > maxProgress) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                {
                    maxProgress = tbProgress;
                    maxProgressTB = tbId;
                }
            }
            if (maxProgressTB != 0xdeaddead)
                attr.currValue = maxProgressTB / attr.bucketSize;
			attr.currRawValue = attr.currValue;
		}
		else if (attr.attrType == SLOW_TB)
		{
            unsigned int minProgress = 0xFFFFFFFF;
            unsigned int minProgressTB = 0xdeaddead;
            attr.currValue = gNumTBsPerSM;

            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                unsigned int tbProgress = gTBProgressArray[index];

                if ((tbProgress < (minProgress - 32)) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                {
                    minProgress = tbProgress;
                    minProgressTB = tbId;
                }
            }
            if (minProgressTB != 0xdeaddead)
                attr.currValue = minProgressTB / attr.bucketSize;
			attr.currRawValue = attr.currValue;
		}
		else if (attr.attrType == TB_WITH_WARPS_FINISHED)
		{
        	unsigned int maxWarpsAtFinish = 0;
        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
            attr.currValue = gNumTBsPerSM;
        	if (lNumWarpsAtFinishMap.size() > 0)
        	{
            	for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            	{
                	unsigned int numWarpsAtFinish = lNumWarpsAtFinishMap[tbId];
                	if (numWarpsAtFinish > maxWarpsAtFinish)
                	{
                    	if (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end())
                    	{
                        	attr.currValue = tbId;
                        	maxWarpsAtFinish = numWarpsAtFinish;
                    	}
                	}
            	}
        	}
			attr.currRawValue = attr.currValue;
		}
		else if (attr.attrType == ANY_TB_WITH_WARPS_FINISHED)
		{
            attr.currValue = 0;

        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
				 iter != lNumWarpsAtFinishMap.end();
				 iter++)
			{
				unsigned int lNumWarpsAtFinish = iter->second;
				if (lNumWarpsAtFinish > 0)
				{
					attr.currValue = 1;
					break;
				}
			}
			attr.currRawValue = attr.currValue;
		}
		else if (attr.attrType == TB_WITH_WARPS_AT_BARRIER)
		{
        	unsigned int maxWarpsAtBarrier = 0;
        	std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
            attr.currValue = gNumTBsPerSM;
        	if (numWarpsAtBarrierMap.size() > 0)
        	{
            	for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            	{
                	unsigned int numWarpsAtBarrier = numWarpsAtBarrierMap[tbId];
                	if (numWarpsAtBarrier > maxWarpsAtBarrier)
                	{
                    	if (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end())
                    	{
                        	attr.currValue = tbId;
                        	maxWarpsAtBarrier = numWarpsAtBarrier;
                    	}
                	}
            	}
        	}
			attr.currRawValue = attr.currValue;
		}
		else if (attr.attrType == ANY_TB_WITH_WARPS_AT_BARRIER)
		{
            attr.currValue = 0;

        	std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = numWarpsAtBarrierMap.begin();
				 iter != numWarpsAtBarrierMap.end();
				 iter++)
			{
				unsigned int lNumWarpsAtBarrier = iter->second;
				if (lNumWarpsAtBarrier > 0)
				{
					attr.currValue = 1;
					break;
				}
			}
			attr.currRawValue = attr.currValue;
		}
        else if (attr.attrType == WHICH_PIPELINE)
        {
            attr.currValue = UNKNOWN_OP;

            if (dRLSched->readyLongLatMemInstrs == true)
            {
                if (rl_scheduler::gNumWarpsExecutingMemInstrGPU > HIGH_NUM_WARPS_EXECUTING_MEM_INSTR)
                {
                    if (dRLSched->numReadySfuInstrs)
                        attr.currValue = SFU__OP;
                    else 
                    {
                        assert (dRLSched->numReadyMemInstrs);
                        attr.currValue = MEM__OP;
                    }
                }
                else
                {
                    assert (dRLSched->numReadyMemInstrs);
                    attr.currValue = MEM__OP;
                }
            }
            else if (rl_scheduler::gSFULongLatInstrReady == true)
            {
                assert (dRLSched->numReadySfuInstrs);
                attr.currValue = SFU__OP;
            }
            else
            {
                if (rl_scheduler::gNumWarpsExecutingMemInstrGPU > HIGH_NUM_WARPS_EXECUTING_MEM_INSTR)
                {
                    if (dRLSched->numReadySfuInstrs)
                        attr.currValue = SFU__OP;
                    else if (dRLSched->numReadyMemInstrs)
                        attr.currValue = MEM__OP;
                    else if (dRLSched->numReadySpInstrs)
                        attr.currValue = SP__OP;
                }
                else
                {
                    if (dRLSched->numReadyMemInstrs)
                        attr.currValue = MEM__OP;
                    else if (dRLSched->numReadySfuInstrs)
                        attr.currValue = SFU__OP;
                    else if (dRLSched->numReadySpInstrs)
                        attr.currValue = SP__OP;
                }
            }
			attr.currRawValue = attr.currValue;
        }
        else if (attr.attrType == WHICH_THREAD_BLOCK)
        {
            setWhichThreadBlock(attr);
			attr.currRawValue = attr.currValue;
        }
        else if (attr.attrType == NUM_OF_MEM_QS_LOADED)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumMemSchedQsLoaded);
        }
        else if (attr.attrType == NUM_OF_MEM_REQS_IN_SCHED_Q)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumReqsInMemSchedQs);
        }
        else if (attr.attrType == NUM_OF_SP_SFU_INSTR_PER_MEM_INSTR_ISSUED)
        {
            if (rl_scheduler::gNumGTCMemInstrIssued > 0)
            {
                unsigned int v1 = (rl_scheduler::gNumSpInstrIssued + rl_scheduler::gNumSfuInstrIssued) / rl_scheduler::gNumGTCMemInstrIssued;
                attr.currValue = v1 / attr.bucketSize;
				attr.mSetBucketizedAttrValue(v1);
            }
            else
			{
                attr.currValue = 0;
				attr.currRawValue = 0;
			}
        }
        else if (attr.attrType == NUM_OF_INSTR_ISSUED_PER_L1_MISS)
        {
            struct cache_sub_stats css;

            dRLSched->m_shader->get_L1D_sub_stats(css);

            if (css.misses != 0)
            {
                unsigned int numInstrsPerL1Miss = rl_scheduler::gNumInstrsIssued[smId] / css.misses;

				attr.mSetBucketizedAttrValue(numInstrsPerL1Miss);
            }
			else
				attr.mSetBucketizedAttrValue(rl_scheduler::gNumInstrsIssued[smId]);
		}
        else if (attr.attrType == AVG_GL_MEM_LAT)
        {
            if (rl_scheduler::gNumGTCMemInstrFinished > 0)
            {
                unsigned int avgLat = rl_scheduler::gNumGTCMemLatencyCycles / rl_scheduler::gNumGTCMemInstrFinished;
                attr.currValue = avgLat / attr.bucketSize;
				attr.mSetBucketizedAttrValue(avgLat);
            }
            else
			{
                attr.currValue = 1;
                attr.currRawValue = DEFAULT_GLOBAL_MEM_LATENCY; //avg mem access latency cycles
			}
        }
        else if (attr.attrType == NUM_OF_MEM_INSTRS_EXECUTING_ON_SM)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumWarpsExecutingMemInstr[smId]);
        }
        else if (attr.attrType == NUM_OF_MEM_INSTRS_EXECUTING_ON_GPU)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumWarpsExecutingMemInstrGPU);
        }
        else if (attr.attrType == NUM_OF_READY_ALU_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyAluInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyMemInstrs);
        }
        else if (attr.attrType == NUM_OF_FUTURE_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numFutureMemInstrs);
        }
        else if (attr.attrType == NUM_OF_FUTURE_SFU_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numFutureSfuInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_SFU_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadySfuInstrs);
        }
        else if (attr.attrType == NUM_OF_SPLIT_WARPS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numSplitWarps);
        }
        else if (attr.attrType == NUM_OF_READY_SP_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadySpInstrs);
        }
        else if (attr.attrType == NUM_OF_WAITING_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numWaitingInstrs);
        }
        else if (attr.attrType == NUM_OF_PIPE_STALLS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numPipeStalls);
        }
        else if (attr.attrType == NUM_OF_MEM_PIPE_STALLS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numMemPipeStalls);
        }
        else if (attr.attrType == NUM_OF_SFU_PIPE_STALLS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numSfuPipeStalls);
        }
        else if (attr.attrType == NUM_OF_SP_PIPE_STALLS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numSpPipeStalls);
        }
        else if (attr.attrType == NUM_OF_IDLE_WARPS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numIdleWarps);
        }
        else if (attr.attrType == NUM_OF_SCHEDULABLE_WARPS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numSchedulableWarps);
        }
        else if (attr.attrType == NUM_OF_READY_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_MEM_INSTRS_WITH_SAME_TB_AS_LAST_MEM_INSTR)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumReadyMemInstrsWithSameTB[smId]);
        }
        else if (attr.attrType == NUM_OF_READY_MEM_INSTRS_WITH_SAME_PC_AS_LAST_MEM_INSTR)
        {
			attr.mSetBucketizedAttrValue(rl_scheduler::gNumReadyMemInstrsWithSamePC[smId]);
        }
        else if (attr.attrType == NO_TB_FINISHED)
        {
            attr.currValue = (rl_scheduler::gNumFinishedTBs == 0) ? 1 : 0; 
            attr.currRawValue = (rl_scheduler::gNumFinishedTBs == 0) ? 1 : 0;
        }
        else if (attr.attrType == TBS_WAITING)
        {
            attr.currValue = dRLSched->m_shader->m_cluster->get_gpu()->get_more_cta_left() ? 1 : 0;
            attr.currRawValue = dRLSched->m_shader->m_cluster->get_gpu()->get_more_cta_left() ? 1 : 0;
        }
        else if (attr.attrType == NUM_OF_WARPS_ISSUED_BARRIER)
        {
			assert(0);
            if (attr.bucketSize == 0)
            {
                if (gNumWarpsPerBlock < 4)
                    attr.bucketSize = gNumWarpsPerBlock;
                else
                    attr.bucketSize = gNumWarpsPerBlock / 4;
            }
            if (attr.bucketSize == 0)
                attr.currValue = attr.defaultValue;
            else
                attr.currValue = rl_scheduler::gNumWarpsWaitingAtBarrier[smId] / attr.bucketSize;
        }
        else if (attr.attrType == NUM_OF_WARPS_FINISHED)
        {
			assert(0);
            if (attr.bucketSize == 0)
            {
                if (gNumWarpsPerBlock < 4)
                    attr.bucketSize = gNumWarpsPerBlock;
                else
                    attr.bucketSize = gNumWarpsPerBlock / 4;
            }
            if (attr.bucketSize == 0)
                attr.currValue = attr.defaultValue;
            else
                attr.currValue = rl_scheduler::gNumWarpsFinished[smId] / attr.bucketSize;
        }
        else if (attr.attrType == NUM_OF_READY_READ_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyReadMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_WRITE_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyWriteMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_GLOBAL_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyGlobalMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_SHARED_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadySharedMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_SHARED_TEX_CONST_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadySharedTexConstMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_CONSTANT_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyConstantMemInstrs);
        }
        else if (attr.attrType == NUM_OF_READY_TEXTURE_MEM_INSTRS)
        {
			attr.mSetBucketizedAttrValue(dRLSched->numReadyTextureMemInstrs);
        }
        else if (attr.attrType == I_CACHE_MISS_PERCENT)
        {
            struct cache_sub_stats css;

            dRLSched->m_shader->get_L1I_sub_stats(css);

            if (css.accesses != 0)
            {
                unsigned int missPercent = (css.misses * 100) / css.accesses;

				attr.mSetBucketizedAttrValue(missPercent);
            }
        }
        else if (attr.attrType == L1_MISS_PERCENT)
        {
            struct cache_sub_stats css;

            dRLSched->m_shader->get_L1D_sub_stats(css);

            if (css.accesses != 0)
            {
                unsigned int missPercent = (css.misses * 100) / css.accesses;

				attr.mSetBucketizedAttrValue(missPercent);
            }
        }
        else if (attr.attrType == L2_MISS_PERCENT)
        {
            unsigned int numAccesses = 0;
            unsigned int numMisses = 0;

            dRLSched->m_shader->m_cluster->get_gpu()->getL2Stats(numAccesses, numMisses);
            if (numAccesses != 0)
            {
                unsigned int missPercent = (numMisses * 100) / numAccesses;

				attr.mSetBucketizedAttrValue(missPercent);
            }
        }
        else if (attr.attrType == READY_SP_INSTRS)
        {
            attr.currValue = dRLSched->readySpInstrs;
        }
        else if (attr.attrType == READY_SFU_INSTRS)
        {
            attr.currValue = dRLSched->readySfuInstrs;
        }
        else if (attr.attrType == READY_MEM_INSTRS)
        {
            attr.currValue = dRLSched->readyMemInstrs;
        }
        else if (attr.attrType == READY_GMEM_INSTRS)
        {
            attr.currValue = dRLSched->readyGlobalMemInstrs;
        }
        else if (attr.attrType == READY_LMEM_INSTRS)
        {
            attr.currValue = dRLSched->readyLongLatMemInstrs;
        }
        else if (attr.attrType == READY_SMEM_INSTRS)
        {
            attr.currValue = dRLSched->readySharedMemInstrs;
        }
        else if (attr.attrType == READY_CMEM_INSTRS)
        {
            attr.currValue = dRLSched->readyConstMemInstrs;
        }
        else if (attr.attrType == READY_TMEM_INSTRS)
        {
            attr.currValue = dRLSched->readyTexMemInstrs;
        }
        else if (attr.attrType == READY_SHARED_TEX_CONST_MEM_INSTRS)
        {
            attr.currValue = dRLSched->readySharedTexConstMemInstr;
        }
        else if (attr.attrType == READY_GLOBAL_CONST_TEXTURE_MEM_INSTRS)
        {
            attr.currValue = dRLSched->readyGlobalConstTexMemInstr;
        }
        else if (attr.attrType == READY_GLOBAL_CONST_TEXTURE_READ_MEM_INSTRS)
        {
            attr.currValue = dRLSched->readyGlobalConstTexReadMemInstr;
        }
        else if (attr.attrType == READY_GLOBAL_READ_MEM_INSTRS)
        {
            attr.currValue = dRLSched->readyGlobalReadMemInstr;
        }
        else 
            assert(0);

        if (attr.currValue >= attr.numAttrValues)
            attr.currValue = attr.numAttrValues - 1;
    }
}

bool rlEngine::setWhichThreadBlock(rl_attribute& attr)
{
    unsigned int smId = dRLSched->m_shader->get_sid();
    bool moreTBsLeft = dRLSched->m_shader->m_cluster->get_gpu()->get_more_cta_left() ? true : false;
    bool attrValSet = false;

    if (moreTBsLeft)
    {
        unsigned int maxWarpsAtFinish = 0;
        std::map<unsigned int, unsigned int>& numWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
        if (numWarpsAtFinishMap.size() > 0)
        {
            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned int numWarpsAtFinish = numWarpsAtFinishMap[tbId];
                if (numWarpsAtFinish > maxWarpsAtFinish)
                {
                    if (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end())
                    {
                        attr.currValue = tbId;
                        attrValSet = true;
                        maxWarpsAtFinish = numWarpsAtFinish;
                    }
                }
            }
        }
    }
    if (attrValSet == false)
    {
        unsigned int maxWarpsAtBarrier = 0;
        std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
        if (numWarpsAtBarrierMap.size() > 0)
        {
            //for (unsigned int tbId = 0; tbId < MAX_NUM_TB_PER_SM; tbId++)
            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned int numWarpsAtBarrier = numWarpsAtBarrierMap[tbId];
                if (numWarpsAtBarrier > maxWarpsAtBarrier)
                {
                    if (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end())
                    {
                        attr.currValue = tbId;
                        attrValSet = true;
                        maxWarpsAtBarrier = numWarpsAtBarrier;
                    }
                }
            }
        }
    }

    if ((attrValSet == false) && gTBProgressArray)
    {
        if (moreTBsLeft)
        {
            unsigned int maxProgress = 0;
            unsigned int maxProgressTB = 0xdeaddead;

            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                unsigned int tbProgress = gTBProgressArray[index];
    
                if ((tbProgress > maxProgress) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                {
                    maxProgress = tbProgress;
                    maxProgressTB = tbId;
                }
            }
            if (maxProgressTB != 0xdeaddead)
            {
                attr.currValue = maxProgressTB / attr.bucketSize;
                attrValSet = true;
            }
        }
        else
        {
            unsigned int minProgress = 0xFFFFFFFF;
            unsigned int minProgressTB = 0xdeaddead;

            for (unsigned int tbId = 0; tbId < gNumTBsPerSM; tbId++)
            {
                unsigned index = smId * MAX_NUM_TB_PER_SM + tbId;
                unsigned int tbProgress = gTBProgressArray[index];

                if ((tbProgress < (minProgress - 32)) && (gReadyTBIdSet.find(tbId) != gReadyTBIdSet.end()))
                {
                    minProgress = tbProgress;
                    minProgressTB = tbId;
                }
            }
            if (minProgressTB != 0xdeaddead)
            {
                attr.currValue = minProgressTB / attr.bucketSize;
                attrValSet = true;
            }
        }
    }
    return attrValSet;
}

void rlEngine::setAttributeValues1(shd_warp_t* warp)
{
    unsigned int smId = dRLSched->m_shader->get_sid();
    assert(smId < gNumSMs);
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];
        if (attr.attrType == LAST_WARP_ISSUED)
        {
            if (warp)
            {
                uint warpId = warp->get_warp_id();
                if ((warpId & 0x1) == 0)
                    attr.currValue = (warpId >> 1) / attr.bucketSize;
                else
                    attr.currValue = ((warpId - 1) >> 1) / attr.bucketSize;
            }
        }
        else if (attr.attrType == LAST_TB_ISSUED)
        {
            if (warp)
                attr.currValue = warp->get_cta_id();
        }
        else if (attr.attrType == WHICH_THREAD_BLOCK)
        {
            bool attrValSet = setWhichThreadBlock(attr);
            if ((attrValSet == false) && warp)
            {
                attr.currValue = warp->get_cta_id();
                attr.currRawValue = warp->get_cta_id();
                attrValSet = true;
            }
        }
    }
}

void rl_scheduler::computeCurrState()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->computeCurrState();
}

unsigned int rl_attribute::mUsefulValuesForAction(unsigned int xAction, unsigned int xActionType)
{
    //L1MP L2MP NRGMI NRI NRAI NRSFI NRSTCMI NRSPI NWI STBRMI SMNMIE NAIPMI

	unsigned int lRetVal = numAttrValues;
/*
    if (xActionType == USE_CMD_PIPE_AS_ACTION)
	{
		if (xAction == SCHED_NO_INSTR)
		{
			if ((attrType == NUM_OF_READY_GLOBAL_MEM_INSTRS) || 
				(attrType == NUM_OF_READY_INSTRS) || 
				(attrType == NUM_OF_READY_ALU_INSTRS) || 
				(attrType == NUM_OF_READY_SFU_INSTRS) || 
				(attrType == NUM_OF_READY_SP_INSTRS) || 
				(attrType == NUM_OF_SPLIT_WARPS) || 
				(attrType == NUM_OF_READY_SHARED_TEX_CONST_MEM_INSTRS) || 
				(attrType == NUM_OF_WAITING_INSTRS) ||
				(attrType == NUM_OF_MEM_PIPE_STALLS) ||
				(attrType == NUM_OF_SFU_PIPE_STALLS) ||
				(attrType == NUM_OF_SP_PIPE_STALLS) ||
				(attrType == NUM_OF_IDLE_WARPS) ||
				(attrType == NUM_OF_READY_MEM_INSTRS_WITH_SAME_TB_AS_LAST_MEM_INSTR) || 
				(attrType == NUM_OF_SP_SFU_INSTR_PER_MEM_INSTR_ISSUED))
			{
				lRetVal = 1;
			}
		}
		else if (xAction == SCHED_SP_INSTR)
		{
			if ((attrType == L1_MISS_PERCENT) ||
				(attrType == L2_MISS_PERCENT) ||
				(attrType == I_CACHE_MISS_PERCENT) ||
				(attrType == NUM_OF_READY_GLOBAL_MEM_INSTRS) || 
				(attrType == NUM_OF_READY_INSTRS) || 
				(attrType == NUM_OF_READY_ALU_INSTRS) || 
				(attrType == NUM_OF_READY_SFU_INSTRS) || 
				(attrType == NUM_OF_READY_SHARED_TEX_CONST_MEM_INSTRS) || 
				(attrType == NUM_OF_WAITING_INSTRS) || 
				(attrType == NUM_OF_MEM_PIPE_STALLS) ||
				(attrType == NUM_OF_SFU_PIPE_STALLS) ||
				(attrType == NUM_OF_MEM_INSTRS_EXECUTING_ON_SM) || 
				(attrType == NUM_OF_READY_MEM_INSTRS_WITH_SAME_TB_AS_LAST_MEM_INSTR) || 
				(attrType == NUM_OF_SP_SFU_INSTR_PER_MEM_INSTR_ISSUED))
			{
				lRetVal = 2;
			}
		}
		else if (xAction == SCHED_SFU_INSTR)
		{
			if ((attrType == L1_MISS_PERCENT) ||
				(attrType == L2_MISS_PERCENT) ||
				(attrType == I_CACHE_MISS_PERCENT) ||
				(attrType == NUM_OF_READY_GLOBAL_MEM_INSTRS) || 
				(attrType == NUM_OF_READY_INSTRS) || 
				(attrType == NUM_OF_READY_ALU_INSTRS) || 
				(attrType == NUM_OF_READY_SP_INSTRS) || 
				(attrType == NUM_OF_SPLIT_WARPS) || 
				(attrType == NUM_OF_READY_SHARED_TEX_CONST_MEM_INSTRS) || 
				(attrType == NUM_OF_WAITING_INSTRS) || 
				(attrType == NUM_OF_MEM_PIPE_STALLS) ||
				(attrType == NUM_OF_SP_PIPE_STALLS) ||
				(attrType == NUM_OF_IDLE_WARPS) ||
				(attrType == NUM_OF_MEM_INSTRS_EXECUTING_ON_SM) || 
				(attrType == NUM_OF_READY_MEM_INSTRS_WITH_SAME_TB_AS_LAST_MEM_INSTR) || 
				(attrType == NUM_OF_SP_SFU_INSTR_PER_MEM_INSTR_ISSUED))
			{
				lRetVal = 2;
			}
		}
		else if (xAction == SCHED_GMEM_INSTR)
		{
			if ((attrType == NUM_OF_READY_INSTRS) || 
				(attrType == NUM_OF_READY_ALU_INSTRS) || 
				(attrType == NUM_OF_READY_SFU_INSTRS) || 
				(attrType == NUM_OF_READY_SP_INSTRS) || 
				(attrType == NUM_OF_SPLIT_WARPS) || 
				(attrType == NUM_OF_SFU_PIPE_STALLS) ||
				(attrType == NUM_OF_SP_PIPE_STALLS) ||
				(attrType == NUM_OF_IDLE_WARPS) ||
				(attrType == NUM_OF_WAITING_INSTRS))
			{
				lRetVal = 2;
			}
		}
		else if (xAction == SCHED_STC_MEM_INSTR)
		{
			if ((attrType == NUM_OF_READY_INSTRS) || 
				(attrType == NUM_OF_READY_ALU_INSTRS) || 
				(attrType == NUM_OF_READY_SFU_INSTRS) || 
				(attrType == NUM_OF_READY_SP_INSTRS) || 
				(attrType == NUM_OF_SPLIT_WARPS) || 
				(attrType == NUM_OF_SFU_PIPE_STALLS) ||
				(attrType == NUM_OF_SP_PIPE_STALLS) ||
				(attrType == NUM_OF_IDLE_WARPS) ||
				(attrType == NUM_OF_WAITING_INSTRS))
			{
				lRetVal = 2;
			}
		}
		else
			assert(0);
	}
*/
	return lRetVal;
}

float rlEngine::mGetQvalueFromFeaturesAndWeights(unsigned long long index)
{
	assert(rl_scheduler::gUseFeatureWeightFuncApprox == true);

	float qValue = 0.0;

	populateStateActionFeatureValueArray(index);

    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;
	for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
	{
		qValue += (dStateActionWeightArray[i] * dStateActionFeatureValueArray[i]);
		if ((isinf(qValue) != 0) || (isnan(qValue) != 0))
		{
			printf("%llu: OUCH qValue is inf/nan, i = %u, dStateActionWeightArray[%u] = %e, dStateActionFeatureValueArray[%u] = %e, sm %u, sched %u engine %u\n", gpu_sim_cycle,  i, i, dStateActionWeightArray[i], i, dStateActionFeatureValueArray[i], smId, schedId, dEngineNum);
			printWeights();
			assert(0);
		}
	}
	return qValue;
}

float rlEngine::mGetHvalueFromFeaturesAndWeights(unsigned long long index)
{
	assert(rl_scheduler::gUseFeatureWeightFuncApprox == true);

	float hValue = 0.0;

	populateStateActionFeatureValueArray(index);

    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;
	for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
	{
		hValue += (dStateActionWeightArray[i] * dStateActionFeatureValueArray[i]);
		if ((isinf(hValue) != 0) || (isnan(hValue) != 0))
		{
			printf("%llu: OUCH hValue is inf/nan, i = %u, dStateActionWeightArray[%u] = %e, dStateActionFeatureValueArray[%u] = %e, sm %u, sched %u engine %u\n", gpu_sim_cycle,  i, i, dStateActionWeightArray[i], i, dStateActionFeatureValueArray[i], smId, schedId, dEngineNum);
			printWeights();
			assert(0);
		}
	}
	return hValue;
}

float rlEngine::mGetSvalueFromFeaturesAndWeights(unsigned long long index)
{
	assert(rl_scheduler::gUseFeatureWeightFuncApprox == true);

	float sValue = 0.0;

	populateStateFeatureValueArray(index);

    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;
	for (unsigned int i = 0; i < dAttributeVector.size(); i++)
	{
		sValue += (dStateWeightArray[i] * dStateFeatureValueArray[i]);
		if ((isinf(sValue) != 0) || (isnan(sValue) != 0))
		{
			printf("%llu: OUCH sValue is inf/nan, i = %u, dStateWeightArray[%u] = %e, dStateFeatureValueArray[%u] = %e, sm %u, sched %u engine %u\n", gpu_sim_cycle,  i, i, dStateWeightArray[i], i, dStateFeatureValueArray[i], smId, schedId, dEngineNum);
			printWeights();
			assert(0);
		}
	}
	return sValue;
}

float* gTmpStateActionFeatureValueArray = 0;
float* gTmpStateFeatureValueArray = 0;

void rlEngine::populateGlobalStateActionFeatureValueArrayForAllStates()
{
	if (gTmpStateActionFeatureValueArray == 0)
	{
		gTmpStateActionFeatureValueArray = new float[dAttributeVector.size() * dNumActions];

		gTmpStateFeatureValueArray = new float[dAttributeVector.size()];

		for (unsigned int j = 0; j < dAttributeVector.size(); j++)
		{
        	rl_attribute& attr = dAttributeVector[j];
			assert((attr.numAttrValues == 2) || (attr.numAttrValues == 4) || (attr.numAttrValues == 8));
			gTmpStateFeatureValueArray[j] = (dNumStates/attr.numAttrValues) * (((1 << attr.numAttrValues) - 1) / (1 << (attr.numAttrValues - 1)));
		}
	}

	unsigned int lBase = dAttributeVector.size() * dPrevAction;
	for (unsigned int j = 0; j < dAttributeVector.size(); j++)
	{
		gTmpStateActionFeatureValueArray[lBase + j] = gTmpStateFeatureValueArray[j];
	}
}

void rlEngine::populateStateActionFeatureValueArray(unsigned long long index)
{
	if (dValues == 0)
		return;

	for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		dStateActionFeatureValueArray[i] = 0.0;

	unsigned long long stateVal = index / dNumActions;
	unsigned int actionVal = index % dNumActions;

	unsigned int lBase = dAttributeVector.size() * actionVal;
	unsigned long long lStateVal = stateVal;
    for (int i = dAttributeVector.size() - 1; i >= 0; i--)
    {
        rl_attribute& attr = dAttributeVector[i];
		unsigned int attrVal = lStateVal / attr.dPlaceValue;
		assert(attrVal < 8);
		assert(attr.numAttrValues <= 8);
		dStateActionFeatureValueArray[lBase + i] = 1.0 / (float)(1 << attrVal);

		lStateVal = lStateVal % attr.dPlaceValue;
    }
}

void rlEngine::populateStateFeatureValueArray(unsigned long long index)
{
	if (dValues == 0)
		return;

	for (unsigned int i = 0; i < dAttributeVector.size(); i++)
		dStateFeatureValueArray[i] = 0.0;

	unsigned long long stateVal = index / dNumActions;

	unsigned long long lStateVal = stateVal;
    for (int i = dAttributeVector.size() - 1; i >= 0; i--)
    {
        rl_attribute& attr = dAttributeVector[i];
		unsigned int attrVal = lStateVal / attr.dPlaceValue;
		assert(attrVal < 8);
		assert(attr.numAttrValues <= 8);
		dStateFeatureValueArray[i] = 1.0 / (float)(1 << attrVal);

		lStateVal = lStateVal % attr.dPlaceValue;
    }
}

unsigned long long rlEngine::getCurrStateForAction(unsigned int xAction)
{
	if (dValues == 0)
		return 0;

	unsigned long long lCurrState = 0;
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];
		assert(attr.dPlaceValue != 0xdeaddead);

        unsigned int lVal;
		unsigned int lNumAttrValuesToUse = attr.mUsefulValuesForAction(xAction, dActionType);
		if (lNumAttrValuesToUse == 1)
			lVal = 0;
		else if (lNumAttrValuesToUse < attr.numAttrValues)
			lVal = attr.mGetBucketValue(lNumAttrValuesToUse);
		else
        	lVal = attr.currValue;
        lCurrState += (lVal * attr.dPlaceValue);
    }
	return lCurrState;
}

void rlEngine::computeCurrState()
{
	if (dValues == 0)
		return;

	unsigned long long lCurrState = 0;
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];
		assert(attr.dPlaceValue != 0xdeaddead);

		std::string attrNameStr(attr.attrName);
		std::map<std::string, std::map<unsigned int, unsigned int> >::iterator iter = gAttrNameValueCntMap.find(attrNameStr);
		if (iter == gAttrNameValueCntMap.end())
		{
			std::map<unsigned int, unsigned int> valCntMap;
			valCntMap[attr.currRawValue] = 1;
			gAttrNameValueCntMap[attrNameStr] = valCntMap;
		}
		else
		{
			std::map<unsigned int, unsigned int>& valCntMap = iter->second;
			std::map<unsigned int, unsigned int>::iterator iter2 = valCntMap.find(attr.currRawValue);
			if (iter2 == valCntMap.end())
				valCntMap[attr.currRawValue] = 0;
			else
				valCntMap[attr.currRawValue]++;
		}

		dCurrStateVector[i] = attr.currRawValue;
        unsigned int lVal = attr.currValue;
        lCurrState += (lVal * attr.dPlaceValue);
    }
	dCurrState = lCurrState;
}

void takePrimaryActionCntSnapshot()
{
	unsigned int snapShotFreq = 1000;
	if (gpu_sim_cycle <= 25000)
		snapShotFreq = 1000;
	else if (gpu_sim_cycle <= 75000)
		snapShotFreq = 2000;
	else if (gpu_sim_cycle <= 175000)
		snapShotFreq = 4000;
	else if (gpu_sim_cycle <= 375000)
		snapShotFreq = 8000;
	else if (gpu_sim_cycle <= 775000)
		snapShotFreq = 16000;
	else if (gpu_sim_cycle <= 1575000)
		snapShotFreq = 32000;
	else
		snapShotFreq = 64000;
	if ((gpu_sim_cycle % snapShotFreq) == 0)
	{
		if (gPrimaryActionCntSnapshotCycle < gpu_sim_cycle)
		{
			gPrimaryActionCntMapVec.push_back(gPrimaryActionCntMap);
			gPrimaryActionCntSnapshotCycle = gpu_sim_cycle;
		}
	}
}

void rlEngine::printPrimaryActionCntSnapshots()
{
	printf("BEGIN primary action count snapshot\n");
	printf("CYCLE");
	for (unsigned int i = 0; i < dNumActions; i++)
	{
		std::string actionStr = getActionStr(i);
		printf(" %s", actionStr.c_str());
	}
	printf("\n");
	
	unsigned int snapShotFreq = 1000;
	unsigned long long snapShotTime = 0;
	for (unsigned int i1 = 0; i1 < gPrimaryActionCntMapVec.size(); i1++)
	{
		if (i1 < 25)
			snapShotFreq = 1000;
		else if (i1 < 75)
			snapShotFreq = 2000;
		else if (i1 < 175)
			snapShotFreq = 4000;
		else if (i1 < 375)
			snapShotFreq = 8000;
		else if (i1 < 775)
			snapShotFreq = 16000;
		else if (i1 < 1575)
			snapShotFreq = 32000;
		else
			snapShotFreq = 64000;
		snapShotTime += snapShotFreq;
		std::map<unsigned int, unsigned int>& lPrimaryActionCntMap = gPrimaryActionCntMapVec[i1];
		printf("%llu", snapShotTime);
		for (unsigned int i = 0; i < dNumActions; i++)
			printf(" %u", lPrimaryActionCntMap[i]);
		printf("\n");
	}
	gPrimaryActionCntMapVec.clear();
	printf("END primary action count snapshot\n");
}

unsigned long long rlEngine::getQvalueMatrixSize()
{
	dNumStates = 1;
    for (size_t i = 0; i < dAttributeVector.size(); i++)
    {
        rl_attribute& attr = dAttributeVector[i];
		attr.dPlaceValue = dNumStates;

		dNumStates *= attr.numAttrValues;
    }

	dNumActions = 0;
    if (dActionType == USE_TB_CMD_PIPE_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_TB_ID_AND_CMD_PIPE;
	}
    else if (dActionType == USE_TB_ID_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_TB_ID;
	}
    else if (dActionType == USE_WARP_ID_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_WARP_ID;
	}
    else if (dActionType == USE_TB_WARP_ID_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_TB_ID_AND_WARP_ID;
	}
    else if (dActionType == USE_TB_TYPE_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_TB_TYPE;
	}
    else if (dActionType == USE_CMD_PIPE_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_CMD_PIPE;
	}
    else if (dActionType == USE_NUM_WARPS_AND_L1_BYPASS_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_NUM_WARPS_AND_BYPASS_L1;
	}
    else if (dActionType == USE_NUM_WARPS_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_NUM_WARPS;
	}
    else if (dActionType == USE_L1_BYPASS_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_BYPASS_L1;
	}
    else if (dActionType == USE_WHICH_SCHED_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_WHICH_SCHED;
	}
    else if (dActionType == USE_WHICH_WARP_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_WHICH_WARP;
	}
    else if (dActionType == USE_WHICH_WARP_TYPE_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE;
	}
    else if (dActionType == USE_LRR_GTO_AS_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_LRR_GTO_TYPE;
	}
    else if (dActionType == USE_NAM_ACTION)
	{
		dNumActions = MAX_ACTIONS_OF_TYPE_NAM;
	}

	gPrimaryActionCntSnapshotCycle = 0;
	for (unsigned int i = 0;  i < dNumActions; i++)
	{
		gPrimaryActionCntMap[i] = 0;
		gSecondaryActionCntMap[i] = 0;
	}

    unsigned long long matrixSize;
	if (gActorCriticMethod)
    	matrixSize = dNumStates;
	else
    	matrixSize = dNumStates * dNumActions;
    uint smId = dRLSched->m_shader->get_sid();
    uint schedId = dRLSched->m_id;
	if ((smId == 0) && (schedId == 0))
		printf("qvalue table size %llu, num states = %llu, num actions = %u\n", matrixSize, dNumStates, dNumActions);

    return matrixSize;
}

void rlEngine::mSetQvalueUpdateCnt(unsigned long long xIndex, unsigned int updateCnt)
{
	dValueUpdates->mSetQvalueUpdateCnt(xIndex, updateCnt);
}

void rlEngine::mSetQvalue(unsigned long long xIndex, float xVal)
{
	dValues->mSetQvalue(xIndex, xVal);
}

void rlEngine::mSetHvalue(unsigned long long xIndex, float xVal)
{
	dValues->mSetHvalue(xIndex, xVal);
}

void rlEngine::mSetSvalue(unsigned long long xIndex, float xVal)
{
	dValues->mSetSvalue(xIndex, xVal);
}

//#define MAX_Q_VALUE_TABLE_SIZE (32 * 1024 * 1024)
//#define MAX_Q_VALUE_TABLE_SIZE_NUM_BITS 25

#define MAX_Q_VALUE_TABLE_SIZE (2 * 1024) //default value
#define MAX_Q_VALUE_TABLE_SIZE_NUM_BITS 11

//#define MAX_Q_VALUE_TABLE_SIZE (1024)
//#define MAX_Q_VALUE_TABLE_SIZE_NUM_BITS 10

//#define MAX_Q_VALUE_TABLE_SIZE (512)
//#define MAX_Q_VALUE_TABLE_SIZE_NUM_BITS 9

unsigned int gQtableSize = MAX_Q_VALUE_TABLE_SIZE;
unsigned int gQtableSizeNumBits = MAX_Q_VALUE_TABLE_SIZE_NUM_BITS;

unsigned int gGetHashedIndex(unsigned long long xIndex)
{
	unsigned int lIndex = 0;
	unsigned long long tmp = xIndex;
	while (tmp)
	{
		unsigned int tmp2 = (tmp & (gQtableSize - 1));
		lIndex = tmp2 ^ lIndex;
		tmp = tmp >> gQtableSizeNumBits;
	}
	return lIndex;
}

unsigned int valueUpdateMap::mGetQvalueUpdate(unsigned long long xIndex)
{
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, unsigned int>::iterator lIter = dMap.find(lIndex);
	float lValue = 0;
	if (lIter != dMap.end())
		lValue = lIter->second;
	return lValue;
}

unsigned int rlEngine::mGetQvalueUpdate(unsigned long long xIndex)
{
	unsigned int lIndex = gGetHashedIndex(xIndex);
	return dValueUpdates->mGetQvalueUpdate(lIndex);
}

void valueUpdateMap::mSetQvalueUpdateCnt(unsigned long long xIndex, unsigned int updateCnt)
{
	unsigned int lIndex = gGetHashedIndex(xIndex);
	dMap[lIndex] = updateCnt;
}

void valueMap::mSetHvalue(unsigned long long xIndex, float xVal)
{
	assert(gActorCriticMethod == true);
	unsigned int lIndex = gGetHashedIndex(xIndex);
	dStateActionValueMap[lIndex] = xVal;
}
void valueMap::mSetSvalue(unsigned long long xIndex, float xVal)
{
	assert(gActorCriticMethod == true);
	unsigned int lIndex = gGetHashedIndex(xIndex);
	dStateValueMap[lIndex] = xVal;
}
void valueMap::mSetQvalue(unsigned long long xIndex, float xVal)
{
	assert(gActorCriticMethod == false);
	unsigned int lIndex = gGetHashedIndex(xIndex);
	dStateActionValueMap[lIndex] = xVal;
}

unsigned int rlEngine::mGetTotalQvalueUpdates(unsigned int& xNumCellsTouched)
{
	return dValueUpdates->mGetTotalQvalueUpdates(xNumCellsTouched);
}

float rlEngine::mGetTotalQvalue()
{
	return dValues->mGetTotalQvalue(dStateActionValueArraySize);
}

float valueMap::mGetTotalQvalue(unsigned int xQvalueArraySize)
{
	float lTotalQvalue = 0.0;
	unsigned int lMaxSize = xQvalueArraySize > gQtableSize ? gQtableSize : xQvalueArraySize;
	for (unsigned long long i = 0; i < lMaxSize; i++)
	{
		unsigned int lIndex = gGetHashedIndex(i);
		std::map<unsigned long long, float>::iterator lIter = dStateActionValueMap.find(lIndex);
		if (lIter != dStateActionValueMap.end())
			lTotalQvalue += lIter->second;
	}
	return lTotalQvalue;
}

unsigned int valueUpdateMap::mGetTotalQvalueUpdates(unsigned int& xNumCellsTouched)
{
	xNumCellsTouched = 0;
	unsigned int lTotalQvalueUpdates = 0.0;
	for (std::map<unsigned long long, unsigned int>::iterator lIter = dMap.begin();
		 lIter != dMap.end();
		 lIter++)
	{
		lTotalQvalueUpdates += lIter->second;
		xNumCellsTouched++;
	}
	return lTotalQvalueUpdates;
}

float valueMap::mGetHvalue(unsigned long long xIndex)
{
	assert(gActorCriticMethod == true);
	assert (rl_scheduler::gUseFeatureWeightFuncApprox == false);

	float lValue;
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, float>::iterator lIter = dStateActionValueMap.find(lIndex);
	if (lIter != dStateActionValueMap.end())
		lValue = lIter->second;
	else
	{
		lValue = 0;
	}
	return lValue;
}

float valueMap::mGetSvalue(unsigned long long xIndex)
{
	assert(gActorCriticMethod == true);
	assert (rl_scheduler::gUseFeatureWeightFuncApprox == false);

	float lValue;
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, float>::iterator lIter = dStateValueMap.find(lIndex);
	if (lIter != dStateValueMap.end())
		lValue = lIter->second;
	else
	{
		lValue = 0;
	}
	return lValue;
}

float valueMap::mGetQvalue(unsigned long long xIndex)
{
	if (gActorCriticMethod)
		return mGetHvalue(xIndex);

	assert (rl_scheduler::gUseFeatureWeightFuncApprox == false);

	float lValue;
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, float>::iterator lIter = dStateActionValueMap.find(lIndex);
	if (lIter != dStateActionValueMap.end())
		lValue = lIter->second;
	else
	{
		lValue = 1 / (1 - gGamma);
	}
	return lValue;
}

float rlEngine::mGetQvalue(unsigned long long xIndex)
{
	if (gActorCriticMethod)
		return mGetHvalue(xIndex);

	float lValue;
	if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
	{
		lValue = mGetQvalueFromFeaturesAndWeights(xIndex);
	}
	else
	{
		unsigned int lIndex = gGetHashedIndex(xIndex);
		lValue = dValues->mGetQvalue(lIndex);
	}
	return lValue;
}

float rlEngine::mGetHvalue(unsigned long long xIndex)
{
	assert(gActorCriticMethod == true);
	float lValue;
	if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
	{
		lValue = mGetHvalueFromFeaturesAndWeights(xIndex);
	}
	else
	{
		lValue = dValues->mGetHvalue(xIndex);
	}
	return lValue;
}

float rlEngine::mGetSvalue(unsigned long long xIndex)
{
	assert(gActorCriticMethod == true);
	float lValue;
	if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
	{
		lValue = mGetSvalueFromFeaturesAndWeights(xIndex);
	}
	else
	{
		lValue = dValues->mGetSvalue(xIndex);
	}
	return lValue;
}

rlEngine::rlEngine(rl_scheduler* xRLSched, unsigned int xEngineNum)
{
	//dPrimaryRLEngine = false;
	//dSecondaryRLEngine = false;
	dEngineNum = xEngineNum;
	dActionType = 0xdeaddead;
	dRLSched = xRLSched;
	dPrevState = 0xFFFFFFFF;
	dPrevAction = 0xdeaddead;
/*
	for (int i = 0; i < MAX_STATE_HISTORY; i++)
	{
		dStateHistory[i] = 0xFFFFFFFF;
		dActionHistory[i] = 0xdeaddead;
		if (i == 0)
			dLambda[i] = LAMBDA;
		else
			dLambda[i] = LAMBDA * dLambda[i-1];
	}
*/
	//dPrevState2 = 0xFFFFFFFF;
	//dPrevAction2 = 0xdeaddead;
	dCurrAction = 0;
	dFirstTime = true;
	dSarsaAgent = 0;
	dCMAC = 0;
	dDiscountFactor = 1.0;
}

void valueMap::mClear()
{
	dStateActionValueMap.clear();
	dStateValueMap.clear();
}

void valueUpdateMap::mClear()
{
	dMap.clear();
}

void rlEngine::mClear()
{
	dPrevState = 0xFFFFFFFF;
	dPrevAction = 0xdeaddead;
	dCurrAction = 0;
	dFirstTime = true;
	if ((rl_scheduler::gUsePrevQvalues == false) && (gShareQvalueTableForAllSMs == false))
	{
		dValues->mClear();
		dValueUpdates->mClear();
	}
	if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
	{
		dPrevQForFA = 0.0;
		assert(dStateActionFeatureValueArray);
		assert(dStateActionWeightArray);
		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			dStateActionFeatureValueArray[i] = 0.0;
			dStateActionWeightArray[i] = 0.0;
		}
		if (gNewActorCriticMethod)
		{
			dPrev_vXphi = 0.0;
			assert(dStateActionWeightArray2);
			for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
				dStateActionWeightArray2[i] = 0.0;
		}
		if (gActorCriticMethod)
		{
			dPrevSForFA = 0.0;
			dPrevHForFA = 0.0;
			assert(dStateFeatureValueArray);
			assert(dStateWeightArray);
			for (unsigned int i = 0; i < dAttributeVector.size(); i++)
			{
				dStateFeatureValueArray[i] = 0.0;
				dStateWeightArray[i] = 0.0;
			}
		}
	}
}

void rlEngine::updateActorCriticValues(float selectedStateValue)
{
    assert(dPrevState != 0xFFFFFFFF);
    assert(dPrevAction != 0xdeaddead);
    unsigned long long index = (dPrevState * dNumActions) + dPrevAction;
    float Hprev = mGetHvalue(index);
    float Sprev = mGetSvalue(dPrevState);
	float delta = dReward + (gGamma * selectedStateValue) - Sprev;
	float Htmp = Hprev + gBeta * delta;
	Hprev = Htmp;
	mSetHvalue(index, Hprev);
	mIncrHvalueUpdate(index);
    float Stmp = (1 - dRLSched->dCurrAlpha) * Sprev + dRLSched->dCurrAlpha * (dReward + gGamma * selectedStateValue);
	Sprev = Stmp;
	mSetSvalue(dPrevState, Sprev);
}

void rlEngine::updateQvalue(float selectedQ)
{
    assert(dPrevState != 0xFFFFFFFF);
    assert(dPrevAction != 0xdeaddead);
    unsigned long long index = (dPrevState * dNumActions) + dPrevAction;
    float Qprev = mGetQvalue(index);

    // unsigned int smId = dRLSched->m_shader->get_sid();
	// unsigned int schedId = dRLSched->m_id;
	// if ((smId == 0) && (schedId == 0))
	// {
		// std::string actionStr = getActionStr(dPrevAction);
		// printf("prev action = %s\n", actionStr.c_str());
	// }

    float Qtmp = (1 - dRLSched->dCurrAlpha) * Qprev + dRLSched->dCurrAlpha * (dReward + gGamma * selectedQ);
	if  (isnan(Qtmp) != 0)
	{
		printf("OUCH, Qtmp is nan\n");
		printf("Qprev = %e, selectedQ = %e, alpha = %e, reward = %e, gamma = %e\n", Qprev, selectedQ, dRLSched->dCurrAlpha, dReward, gGamma);
    	float t1 = (1 - dRLSched->dCurrAlpha) * Qprev;
    	float t2 = (dReward + gGamma * selectedQ);
    	float t3 = dRLSched->dCurrAlpha * (dReward + gGamma * selectedQ);
    	float t4 = (1 - dRLSched->dCurrAlpha) * Qprev + dRLSched->dCurrAlpha * (dReward + gGamma * selectedQ);
		printf("t1 = %e\n", t1);
		printf("t2 = %e\n", t2);
		printf("t3 = %e\n", t3);
		printf("t4 = %e\n", t4);
		assert (0);
	}
	Qprev = Qtmp;

    mSetQvalue(index, Qprev);
	mIncrQvalueUpdate(index);

	if (gStateActionUpdateCntMapEnabled)
	{
		if (gStateActionUpdateCntMap.find(dPrevState) == gStateActionUpdateCntMap.end())
		{
			std::vector<unsigned int> actionUpdateVec;
			for (unsigned int i = 0; i < dNumActions; i++)
				actionUpdateVec.push_back(0);
						
			actionUpdateVec[dPrevAction]++;
			gStateActionUpdateCntMap[dPrevState] = actionUpdateVec;
		}
		else
		{
			std::vector<unsigned int>& actionUpdateVec = gStateActionUpdateCntMap[dPrevState];
			actionUpdateVec[dPrevAction]++;
		}
	}
}

void populateHadamardMatrix(int* hadamardMatrix, int rows)
{
	if (rows > 2)
	{
		populateHadamardMatrix(hadamardMatrix, rows/2);
		int* prevHM = new int[(rows/2) * (rows/2)];
		for (int i = 0; i < rows/2; i++)
		{
			for (int j = 0; j < (rows/2); j++)
			{
				prevHM[i * (rows/2) + j] = hadamardMatrix[i * (rows/2) + j];
			}
		}

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < (rows/2); j++)
			{
				for (int k = 0; k < (rows/2); k++)
				{
					int r = j + (i/2) * (rows/2);
					int c = k + (i%2) * (rows/2);
					int lIdx = r * rows + c;
					int rIdx = j * (rows/2) + k;
					if (i < 3)
						hadamardMatrix[lIdx] = prevHM[rIdx];
					else
						hadamardMatrix[lIdx] = -prevHM[rIdx];
				}
			}
		}
		delete prevHM;
	}
	else
	{
		assert(rows == 2);

		hadamardMatrix[0] = 1;
		hadamardMatrix[1] = 1;
		hadamardMatrix[2] = 1;
		hadamardMatrix[3] = -1;
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < rows; j++)
			printf("%2d ", hadamardMatrix[i * rows + j]);

		printf("\n");
	}
	printf("=============\n");
}

int* gHadamardMatrix = 0;
int getHadamardValue(unsigned int i, unsigned int n, unsigned int size)
{
	unsigned int tmp = size + 1;
	float tmp1 = log(tmp) / log(2);
	unsigned int tmp2 = ceil(tmp1);
	unsigned int numRows = 1 << tmp2;
	unsigned int numCols = 1 << tmp2;

	if (gHadamardMatrix == 0)
	{
		gHadamardMatrix = new int[numRows * numCols];
		populateHadamardMatrix(gHadamardMatrix, numRows);
	}
	int row = n % numRows;
	int retVal = gHadamardMatrix[row * numCols + i];
	//printf("hm[%d, %d] = %d\n", n, i, retVal);
	return retVal;
}

void rlEngine::updateStateActionWeightValues(float selectedQ)
{
	assert (rl_scheduler::gUseFeatureWeightFuncApprox == true);

    assert(dPrevState != 0xFFFFFFFF);
    assert(dPrevAction != 0xdeaddead);

    unsigned long long index = (dPrevState * dNumActions) + dPrevAction;

    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;

	populateStateActionFeatureValueArray(index);

	if (gNewActorCriticMethod)
	{
		float vXphi = 0.0;
		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			vXphi += (dStateActionWeightArray2[i] * dStateActionFeatureValueArray[i]);

			if ((isinf(vXphi) != 0) || (isnan(vXphi) != 0))
			{
				printf("%llu: OUCH vXphi is inf/nan, i = %u, dStateActionWeightArray2[%u] = %e, dStateActionFeatureValueArray[%u] = %e, sm %u, sched %u engine %u\n", gpu_sim_cycle,  i, i, dStateActionWeightArray2[i], i, dStateActionFeatureValueArray[i], smId, schedId, dEngineNum);
				printWeights();
				assert(0);
			}
		}

		float delta = dReward + gGamma * vXphi - dPrev_vXphi;

		float aRate = gpu_sim_cycle;
		aRate = pow(aRate, 0.6);
		aRate = 1.0 / aRate;

		float bRate = 1.0 / (float)gpu_sim_cycle;
		float beta = (1.0 / dNumStates);

		//this is for theta in the algorithm
		/*
		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			float currVal = dStateActionWeightArray[i];
			float sum = 0;
			int hadamard = getHadamardValue(i, gpu_sim_cycle, dAttributeVector.size() * dNumActions);
			//for (unsigned int j = 0; j < (dNumStates * dNumActions); j++)
			for (unsigned int j = 0; j < dNumStates; j++)
			{
				populateStateActionFeatureValueArray(j * dPrevAction);

				float tmp_vPhiX = 0.0;
				//for (unsigned int k = 0; k < (dAttributeVector.size() * dNumActions); k++)
				for (unsigned int k = 0; k < dAttributeVector.size(); k++)
				{
					unsigned int idx = dAttributeVector.size() * dPrevAction + k;
					tmp_vPhiX += (dStateActionWeightArray2[idx] * dStateActionFeatureValueArray[idx]);
				}
				float term = tmp_vPhiX / (0.1 * hadamard);
				sum += (term * beta);
			}
			dStateActionWeightArray[i] = currVal + bRate * sum;
		}
		*/

		//iterate over all states and add the phi vectors of each state into global phi vector
		populateGlobalStateActionFeatureValueArrayForAllStates();

		float tmp_vPhiX = 0.0;
		for (unsigned int k = 0; k < dAttributeVector.size(); k++)
		{
			unsigned int idx = dAttributeVector.size() * dPrevAction + k;
			tmp_vPhiX += (dStateActionWeightArray2[idx] * gTmpStateActionFeatureValueArray[idx]);
		}
		float tmp1 = (bRate * beta * tmp_vPhiX) / 0.1;

		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			float currVal = dStateActionWeightArray[i];
			int hadamard = getHadamardValue(i, gpu_sim_cycle, dAttributeVector.size() * dNumActions);

			float newVal = currVal + tmp1 / hadamard;

			if (newVal > gMaxProjVal)
				newVal = gMaxProjVal;
			else if (newVal < gMinProjVal)
				newVal = gMinProjVal;
			else if ((isinf(newVal) != 0) || (isnan(newVal) != 0))
				newVal = gMaxProjVal;
			dStateActionWeightArray[i] = newVal;

			/*
			if ((smId == 0) && (schedId == 0))
			{
				if ((gpu_sim_cycle % 10) == 0)
					printf("%llu: theta %u = %f\n", gpu_sim_cycle, i, dStateActionWeightArray[i]);
			}
			*/
		}

		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			float tmp = aRate * delta;
			float currVal = dStateActionWeightArray2[i];
			if (dStateActionFeatureValueArray[i] != 0)
			{
				float newVal = currVal + tmp * dStateActionFeatureValueArray[i];
				if (newVal > gMaxProjVal)
					newVal = gMaxProjVal;
				else if (newVal < gMinProjVal)
					newVal = gMinProjVal;
				else if ((isinf(newVal) != 0) || (isnan(newVal) != 0))
					newVal = gMaxProjVal;

				dStateActionWeightArray2[i] = newVal;
			}
		}

		dPrev_vXphi = vXphi;
	}
	else
	{
		float delta = dReward + (gGamma * selectedQ) - dPrevQForFA;
	
    	mSetQvalue(index, dPrevQForFA);
		mIncrQvalueUpdate(index);
	
		dPrevQForFA = selectedQ;
	
		float learningRate = dRLSched->dCurrAlpha;
		//float learningRate = 1.0 / (dAttributeVector.size() * dNumActions);
		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
		{
			float currVal = dStateActionWeightArray[i];
	
			float newVal = currVal + learningRate * delta * dStateActionFeatureValueArray[i];
			if (newVal > gMaxProjVal)
				newVal = gMaxProjVal;
			else if (newVal < gMinProjVal)
				newVal = gMinProjVal;
			else if ((isinf(newVal) != 0) || (isnan(newVal) != 0))
				newVal = gMaxProjVal;
			dStateActionWeightArray[i] = newVal;
	
			if ((isinf(dStateActionWeightArray[i]) != 0) || 
		    	(isnan(dStateActionWeightArray[i]) != 0))
			{
				printf("%llu: OUCH dStateActionWeightArray[%u] is inf/nan, sm %u, sched %u, engine %u\n", gpu_sim_cycle, i, smId, schedId, dEngineNum);
				printWeights();
				assert(0);
			}
		}
	}
}

void rlEngine::updateStateActionWeightValuesActorCritic(float selectedV)
{
	assert (rl_scheduler::gUseFeatureWeightFuncApprox == true);

    assert(dPrevState != 0xFFFFFFFF);
    assert(dPrevAction != 0xdeaddead);

    unsigned long long index = (dPrevState * dNumActions) + dPrevAction;
	float delta = dReward + (gGamma * selectedV) - dPrevSForFA;

    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;
    mSetHvalue(index, dPrevHForFA);
	mIncrQvalueUpdate(index);

    unsigned long long curIndex = (dCurrState * dNumActions) + dCurrAction;
	dPrevHForFA = mGetHvalue(curIndex);
	dPrevSForFA = selectedV;

	populateStateActionFeatureValueArray(index);

	//float learningRate = 1.0 / (dAttributeVector.size() * dNumActions);
	float learningRate = dRLSched->dCurrAlpha;
	for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
	{
		float currVal = dStateActionWeightArray[i];

		float newVal = currVal + learningRate * delta * dStateActionFeatureValueArray[i];
		if (newVal > gMaxProjVal)
			newVal = gMaxProjVal;
		else if (newVal < gMinProjVal)
			newVal = gMinProjVal;
		else if ((isinf(newVal) != 0) || (isnan(newVal) != 0))
			newVal = gMaxProjVal;
		dStateActionWeightArray[i] = newVal;

		if ((isinf(dStateActionWeightArray[i]) != 0) || 
		    (isnan(dStateActionWeightArray[i]) != 0) /*|| (dStateActionWeightArray[i] > 1000)*/)
		{
			printf("%llu: OUCH dStateActionWeightArray[%u] is inf/nan, sm %u, sched %u, engine %u\n", gpu_sim_cycle, i, smId, schedId, dEngineNum);
			printWeights();
			assert(0);
		}
	}

	populateStateFeatureValueArray(index);
	//learningRate = 1.0 / dAttributeVector.size();
	learningRate = dRLSched->dCurrAlpha;
	for (unsigned int i = 0; i < dAttributeVector.size(); i++)
	{
		float currVal = dStateWeightArray[i];

		float newVal = currVal + learningRate * delta * dStateFeatureValueArray[i];
		if (newVal > gMaxProjVal)
			newVal = gMaxProjVal;
		else if (newVal < gMinProjVal)
			newVal = gMinProjVal;
		else if ((isinf(newVal) != 0) || (isnan(newVal) != 0))
			newVal = gMaxProjVal;
		dStateWeightArray[i] = newVal;

		if ((isinf(dStateWeightArray[i]) != 0) || 
		    (isnan(dStateWeightArray[i]) != 0) /*|| (dStateWeightArray[i] > 1000)*/)
		{
			printf("%llu: OUCH dStateWeightArray[%u] is inf/nan, sm %u, sched %u, engine %u\n", gpu_sim_cycle, i, smId, schedId, dEngineNum);
			printWeights();
			assert(0);
		}
	}
}

void rlEngine::mIncrQvalueUpdate(unsigned long long xIndex)
{
	/*
    unsigelse ned int smId = dRLSched->m_shader->get_sid();
	if ((smId == 0) && (dEngineNum == 0))
	{
		unsigned int lIndex = gGetHashedIndex(xIndex);
		printf("%llu: xIndex = %llu, hashed index = %u\n", gpu_sim_cycle, xIndex, lIndex);
	}
	*/
	dValueUpdates->mIncrQvalueUpdate(xIndex);
}

void valueUpdateMap::mIncrQvalueUpdate(unsigned long long xIndex)
{
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, unsigned int>::iterator lIter = dMap.find(lIndex);
	if (lIter == dMap.end())
		dMap[lIndex] = 1;
	else
		lIter->second++;
}

void rlEngine::mIncrHvalueUpdate(unsigned long long xIndex)
{
	dValueUpdates->mIncrHvalueUpdate(xIndex);
}

void valueUpdateMap::mIncrHvalueUpdate(unsigned long long xIndex)
{
	unsigned int lIndex = gGetHashedIndex(xIndex);
	std::map<unsigned long long, unsigned int>::iterator lIter = dMap.find(lIndex);
	if (lIter == dMap.end())
		dMap[lIndex] = 1;
	else
		lIter->second++;
}

void rlEngine::addAttribute(char* shortName, char* name, rl_attr_type type, unsigned int numValues, unsigned int defVal, unsigned int bucketSize)
{
	addAttribute(shortName, name, type, numValues, defVal, bucketSize, 0xdeaddead);
}

void rlEngine::addAttribute(char* shortName, char* name, rl_attr_type type, unsigned int numValues, unsigned int defVal, unsigned int bucketSize,
unsigned int xDecreasingBucketSizes)
{
	std::string lNameStr(name);
	if (gAttrNamesSet.find(lNameStr) != gAttrNamesSet.end())
		return;
    if ((dRLSched->m_id == 0) && (dRLSched->m_shader->get_sid() == 0))
        printf("Adding attribute %s, def val %u, num values %u\n", name, defVal, numValues);
	gAttrNamesSet.insert(lNameStr);
    rl_attribute attr(shortName, name, type, numValues, defVal, bucketSize);
	attr.dDecreasingBucketSizes = xDecreasingBucketSizes;
    dAttributeVector.push_back(attr);

	//func approximation (CMAC) uses the following 3 data members
	//pass the original number of values e.g. NUM_OF_READY_SP_INSTRS has 25 values 0-24
	dAttrRange.push_back((double)numValues * bucketSize);
	dAttrMinValue.push_back((double)0.0);
	dAttrResolution.push_back((double)bucketSize);
}

void rlEngine::mCreateCMAC()
{
	int lNumFeatures = dAttributeVector.size();

	double* lRanges = new double[lNumFeatures];
	double* lMinValues = new double[lNumFeatures];
	double* lResoultion = new double[lNumFeatures];

	for (int i = 0; i < lNumFeatures; i++)
	{
		lRanges[i] = dAttrRange[i];
		lMinValues[i] = dAttrMinValue[i];
		lResoultion[i] = dAttrResolution[i];
	}

	dCMAC = new CMAC(lNumFeatures, dNumActions, lRanges, lMinValues, lResoultion);

	delete lRanges;
	delete lMinValues;
	delete lResoultion;
}

SarsaAgent* rlEngine::mCreateSarsaAgent(SarsaAgent* xSarsaAgent)
{
	if (xSarsaAgent)
		dSarsaAgent = xSarsaAgent;
	else
	{
		int lNumFeatures = dAttributeVector.size();
	
		mCreateCMAC();

    	unsigned int smId = dRLSched->m_shader->get_sid();
		unsigned int schedId = dRLSched->m_id;
		if ((smId == 0) && (schedId == 0))
		{
			printf("num features = %u, num actions = %u, ALPH = %f, XPL = %f\n", lNumFeatures, dNumActions, dRLSched->dCurrAlpha, (dRLSched->dCurrExplorationPercent) / 100.0);
		}

		dSarsaAgent = new SarsaAgent(lNumFeatures, dNumActions, dRLSched->dCurrAlpha, 
                                 	(dRLSched->dCurrExplorationPercent) / 100.0, dCMAC, (char*)"", (char*)"");
	}
	return dSarsaAgent;
}

unsigned int getBucketSize(unsigned int xRange, unsigned int xNumBuckets)
{
	unsigned int lBucketSize = xRange / xNumBuckets;
	if (xRange % xNumBuckets)
		lBucketSize++;
	return lBucketSize;
}

char* gBmName = 0;
char* gKernelName = 0;

void gSetBaseSched()
{
	gBaseSched = 0; //GTO
	if (strcmp(gBmName, "p_sad") == 0)
	{
		if (strcmp(gKernelName, "_Z17larger_sad_calc_8Ptii") == 0)
			gBaseSched = 1; //LRR
	}
	else if (strcmp(gBmName, "LPS") == 0)
		gBaseSched = 1; //LRR
	else if (strcmp(gBmName, "histogram") == 0)
	{
		if ((strcmp(gKernelName, "_Z22mergeHistogram64KernelPjS_j") == 0) || 
		    (strcmp(gKernelName, "_Z18histogram256KernelPjS_j") == 0) || 
		    (strcmp(gKernelName, "_Z23mergeHistogram256KernelPjS_j") == 0))
		{
			gBaseSched = 1; //LRR
		}
	}
	else if (strcmp(gBmName, "MonteCarlo") == 0)
	{
		if (strcmp(gKernelName, "_Z27MonteCarloOneBlockPerOptionPfi") == 0)
			gBaseSched = 1; //LRR
	}
	else if (strcmp(gBmName, "p_sad") == 0)
	{
		if (strcmp(gKernelName, "_Z17larger_sad_calc_8Ptii") == 0)
			gBaseSched = 1; //LRR
	}
	else if (strcmp(gBmName, "p_stencil") == 0)
	{
		if (strcmp(gKernelName, "_Z24block2D_hybrid_coarsen_xffPfS_iii") == 0)
			gBaseSched = 1; //LRR
	}
	else if (strcmp(gBmName, "r_cfd") == 0)
	{
		if ((strcmp(gKernelName, "_Z24cuda_compute_step_factoriPfS_S_") == 0) ||
		    (strcmp(gKernelName, "_Z17cuda_compute_fluxiPiPfS0_S0_") == 0))
		{
			gBaseSched = 1; //LRR
		}
	}
	else if (strcmp(gBmName, "r_lud") == 0)
	{
		if (strcmp(gKernelName, "_Z13lud_perimeterPfii") == 0)
			gBaseSched = 1; //LRR
	}
	else if (strcmp(gBmName, "r_srad_v2") == 0)
	{
		if (strcmp(gKernelName, "_Z11srad_cuda_2PfS_S_S_S_S_iiff") == 0)
			gBaseSched = 1; //LRR
	}
	printf("gBmName = %s, gKernelName = %s, gBaseSched = %u\n", gBmName, gKernelName, gBaseSched);
}

void rlEngine::addAttributes(std::string xAttrStr)
{
	gAttrNamesSet.clear();

    bool warpIdAttr = false;
    bool cmdPipeAttr = false;
    bool tbIdAttr = false;
	bool tbTypeAttr = false;
	bool lTBWSeen = false;
	bool lNTFSeen = false;

	bool lAddOnlyCmdPipeAttr = false;
	bool lAddOnlyTbTypeAttr = false;

	//if (dPrimaryRLEngine)
	if (dEngineNum == 0)
	{
		//this is the primary engine
		if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
			lAddOnlyCmdPipeAttr = true;
		if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
			lAddOnlyTbTypeAttr = true;
	}

	//if (dSecondaryRLEngine)
	if (dEngineNum == 1)
	{
		//this is the secondary engine
		if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
			lAddOnlyTbTypeAttr = true;
		if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
			lAddOnlyCmdPipeAttr = true;
	}

	char lAttrName[64];

    gNumSMs = dRLSched->m_shader->get_config()->n_simt_clusters * dRLSched->m_shader->get_config()->n_simt_cores_per_cluster;

	static bool sPrintFlag = true;

	char lAttrsCharStr[1024];
	strcpy(lAttrsCharStr, xAttrStr.c_str());
	char* lToken = strtok(lAttrsCharStr, "_");
	while (lToken != NULL)
	{
	    if ((lAddOnlyCmdPipeAttr == false) && (lAddOnlyTbTypeAttr == false))
		{
	    	if (strcmp("LW", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"LastWarpIssued", LAST_WARP_ISSUED, 24, 0, 1); //1-47 OR 2-48 in steps of 2
	        	warpIdAttr = true;
	    	}
	    	else if (strcmp("LTB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"LastTBIssued", LAST_TB_ISSUED, gNumTBsPerSM + 1, gNumTBsPerSM, 1); 
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("FTB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"FastTB", FAST_TB, gNumTBsPerSM + 1, gNumTBsPerSM, 1); 
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("STB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"SlowTB", SLOW_TB, gNumTBsPerSM + 1, gNumTBsPerSM, 1); 
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("TBWB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"TBWithWarpsAtBarrier", TB_WITH_WARPS_AT_BARRIER, gNumTBsPerSM + 1, gNumTBsPerSM, 1); 
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("TBWF", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"TBWitWarpsFinished", TB_WITH_WARPS_FINISHED, gNumTBsPerSM + 1, gNumTBsPerSM, 1); 
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("NWIB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"NumOfWarpsIssuedBarrier", NUM_OF_WARPS_ISSUED_BARRIER, 4, 0, gNumWarpsPerBlock / 4);
	        	if (rl_scheduler::gNumWarpsWaitingAtBarrier == 0)
	        	{
	            	rl_scheduler::gNumWarpsWaitingAtBarrier = new unsigned int[gNumSMs];
	            	for (unsigned int i = 0; i < gNumSMs; i++)
	                	rl_scheduler::gNumWarpsWaitingAtBarrier[i] = 0;
	        	}
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("NWF", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"NumOfWarpsFinished", NUM_OF_WARPS_FINISHED, 4, 0, gNumWarpsPerBlock / 4);
	        	if (rl_scheduler::gNumWarpsFinished == 0)
	        	{
	            	rl_scheduler::gNumWarpsFinished = new unsigned int[gNumSMs];
	            	for (unsigned int i = 0; i < gNumSMs; i++)
	                	rl_scheduler::gNumWarpsFinished[i] = 0;
	        	}
	        	tbIdAttr = true;
	    	}
	    	else if (strcmp("WTB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"WhichThreadBlock", WHICH_THREAD_BLOCK, gNumTBsPerSM + 1, gNumTBsPerSM, 1); //1-8
	        	tbIdAttr = true;
		
	        	if (rl_scheduler::gTBNumSpInstrsArray == 0)
	        	{
	            	rl_scheduler::gTBNumSpInstrsArray = new unsigned int[gNumSMs * MAX_NUM_TB_PER_SM];
	            	rl_scheduler::gTBNumSfuInstrsArray = new unsigned int[gNumSMs * MAX_NUM_TB_PER_SM];
	            	rl_scheduler::gTBNumMemInstrsArray = new unsigned int[gNumSMs * MAX_NUM_TB_PER_SM];
	            	for (unsigned int i = 0; i < gNumSMs; i++)
	            	{
	                	for (unsigned int j = 0; j < MAX_NUM_TB_PER_SM; j++)
	                	{
	                    	unsigned int index = i * MAX_NUM_TB_PER_SM + j;
	                    	rl_scheduler::gTBNumSpInstrsArray[index] = 0;
	                    	rl_scheduler::gTBNumSfuInstrsArray[index] = 0;
	                    	rl_scheduler::gTBNumMemInstrsArray[index] = 0;
	                	}
	            	}
	        	}
	    	}
		}
	
		if (strcmp("NTF", lToken) == 0)
		{
			strcpy(lAttrName, "NoThreadBlockFinished");
			addAttribute(lToken, lAttrName, NO_TB_FINISHED, 2, 1, 1);
			lNTFSeen = true;
		}
	    else if (strcmp("TBW", lToken) == 0)
	    {
	       	addAttribute(lToken, (char*)"TBsWaiting", TBS_WAITING, 2, 1, 1);
			lTBWSeen = true;
	    }

		unsigned int lNumBuckets = DEFAULT_NUM_BUCKETS;
	    if (lAddOnlyCmdPipeAttr == false)
		{
	    	if (strcmp("ATBWB", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"AnyTBWithWarpsAtBarrier", ANY_TB_WITH_WARPS_AT_BARRIER, 2, 0, 1); 
	        	tbTypeAttr = true;
	    	}
	    	else if (strcmp("ATBWF", lToken) == 0)
	    	{
	        	addAttribute(lToken, (char*)"AnyTBWitWarpsFinished", ANY_TB_WITH_WARPS_FINISHED, 2, 0, 1); 
	        	tbTypeAttr = true;
	    	}
		}
	
	    if (lAddOnlyTbTypeAttr == false)
		{
			if (checkAttr(lToken, "SMNMIE", lNumBuckets) == true) //if (strcmp("SMNMIE", lToken) == 0)
		    {
				strcpy(lAttrName, "NumOfMemInstrsExecutingOnSM");
		        addAttribute(lToken, lAttrName, NUM_OF_MEM_INSTRS_EXECUTING_ON_SM, lNumBuckets, 0, getBucketSize(40, lNumBuckets), 1);
				allocateNumWarpsExecutingMemInstrArr();
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "GNMIE", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfMemInstrsExecutingOnGPU");
		        addAttribute(lToken, lAttrName, NUM_OF_MEM_INSTRS_EXECUTING_ON_GPU, lNumBuckets, 0, getBucketSize(600, lNumBuckets), 1);
		        rl_scheduler::gNumWarpsExecutingMemInstrGPU = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "STBRMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyMemInstrsWithSameTBasLastMemInstr", NUM_OF_READY_MEM_INSTRS_WITH_SAME_TB_AS_LAST_MEM_INSTR, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
				allocateNumReadyMemInstrsWithSameTBArr();
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRMI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfReadyMemInstrs");
				addAttribute(lToken, lAttrName, NUM_OF_READY_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NFMI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfFutureMemInstrs");
		        addAttribute(lToken, lAttrName, NUM_OF_FUTURE_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numFutureMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NFSFI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfFutureSfuInstrs");
		        addAttribute(lToken, lAttrName, NUM_OF_FUTURE_SFU_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numFutureSfuInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRAI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfReadyAluInstrs");
				addAttribute(lToken, lAttrName, NUM_OF_READY_ALU_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyAluInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRSFI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfReadySfuInstrs");
		        addAttribute(lToken, lAttrName, NUM_OF_READY_SFU_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadySfuInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRSPI", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfReadySpInstrs");
		        addAttribute(lToken, lAttrName, NUM_OF_READY_SP_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadySpInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NSW", lNumBuckets) == true)
		    {
				strcpy(lAttrName, "NumOfSplitWarps");
		        addAttribute(lToken, lAttrName, NUM_OF_SPLIT_WARPS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numSplitWarps = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NMRQ", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumMemReqsInSchedQ", NUM_OF_MEM_REQS_IN_SCHED_Q, lNumBuckets, 0, getBucketSize(96, lNumBuckets), 1);
		        rl_scheduler::gNumReqsInMemSchedQs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NMQL", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumMemQsLoaded", NUM_OF_MEM_QS_LOADED, lNumBuckets, 0, getBucketSize(6, lNumBuckets), 1);
		        rl_scheduler::gNumMemSchedQsLoaded = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NAIPMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfSpSfuInstrPerMemInstrIssued", NUM_OF_SP_SFU_INSTR_PER_MEM_INSTR_ISSUED, lNumBuckets, 0, getBucketSize(20, lNumBuckets), 1);
		        rl_scheduler::gNumSpInstrIssued = 0;
		        rl_scheduler::gNumSfuInstrIssued = 0;
		        rl_scheduler::gNumGTCMemInstrIssued = 0;
		        rl_scheduler::gNumSpInstrIssued1 = 0;
		        rl_scheduler::gNumSfuInstrIssued1 = 0;
		        rl_scheduler::gNumGTCMemInstrIssued1 = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NIPL1M", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfInstrsIssuedPerL1Miss", NUM_OF_INSTR_ISSUED_PER_L1_MISS, lNumBuckets, 0, getBucketSize(100, lNumBuckets), 1);
		        cmdPipeAttr = true;
				allocateNumInstrsIssuedArr();
		    }
		    else if (checkAttr(lToken, "AGML", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"AvgGlobalMemLatency", AVG_GL_MEM_LAT, lNumBuckets, 0, getBucketSize(800, lNumBuckets), 1);
		        rl_scheduler::gNumGTCMemInstrFinished = 0;
		        rl_scheduler::gNumGTCMemLatencyCycles = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NWI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfWaitingInstrs", NUM_OF_WAITING_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numWaitingInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NPS", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfPipeStalls", NUM_OF_PIPE_STALLS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numPipeStalls = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NMPS", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfMemPipeStalls", NUM_OF_MEM_PIPE_STALLS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numMemPipeStalls = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NSFPS", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfSfuPipeStalls", NUM_OF_SFU_PIPE_STALLS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numSfuPipeStalls = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NSPPS", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfSpPipeStalls", NUM_OF_SP_PIPE_STALLS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numSpPipeStalls = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NIW", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfIdleWarps", NUM_OF_IDLE_WARPS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numIdleWarps = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NWS", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfSchedulableWarps", NUM_OF_SCHEDULABLE_WARPS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numSchedulableWarps = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyInstrs", NUM_OF_READY_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "SPCRMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyMemInstrsWithSamePCasLastMemInstr", NUM_OF_READY_MEM_INSTRS_WITH_SAME_PC_AS_LAST_MEM_INSTR, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
				allocateNumReadyMemInstrsWithSamePCArr();
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRRMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyReadMemInstrs", NUM_OF_READY_READ_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyReadMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRWMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyWriteMemInstrs", NUM_OF_READY_WRITE_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyWriteMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRGMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyGlobalMemInstrs", NUM_OF_READY_GLOBAL_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyGlobalMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRSMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadySharedMemInstrs", NUM_OF_READY_SHARED_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadySharedMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRSTCMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadySharedTexConstMemInstrs", NUM_OF_READY_SHARED_TEX_CONST_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadySharedTexConstMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRCMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyConstantMemInstrs", NUM_OF_READY_CONSTANT_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyConstantMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "NRTMI", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"NumOfReadyTextureMemInstrs", NUM_OF_READY_TEXTURE_MEM_INSTRS, lNumBuckets, 0, getBucketSize(24, lNumBuckets), 0);
		        dRLSched->numReadyTextureMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "ICMP", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"ICacheMissPercent", I_CACHE_MISS_PERCENT, lNumBuckets, 0, getBucketSize(100, lNumBuckets), 1);
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "L1MP", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"L1MissPercent", L1_MISS_PERCENT, lNumBuckets, 0, getBucketSize(100, lNumBuckets), 1);
		        cmdPipeAttr = true;
		    }
		    else if (checkAttr(lToken, "L2MP", lNumBuckets) == true)
		    {
		        addAttribute(lToken, (char*)"L2MissPercent", L2_MISS_PERCENT, lNumBuckets, 0, getBucketSize(100, lNumBuckets), 1);
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RSPI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadySpInstrs", READY_SP_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readySpInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RSFI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadySfuInstrs", READY_SFU_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readySfuInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyMemInstrs", READY_MEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RGMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyGlobalMemInstrs", READY_GMEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyGlobalMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RLMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyLongLatMemInstrs", READY_LMEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyLongLatMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RSMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadySharedMemInstrs", READY_SMEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readySharedMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RCMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyConstMemInstrs", READY_CMEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyConstMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RTMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyTexMemInstrs", READY_TMEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyTexMemInstrs = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RSTCMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadySharedTexConstMemInstrs", READY_SHARED_TEX_CONST_MEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readySharedTexConstMemInstr = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RGCTRMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyGlobalConstTextureReadMemInstrs", READY_GLOBAL_CONST_TEXTURE_READ_MEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyGlobalConstTexReadMemInstr = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RGCTMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyGlobalConstTextureMemInstrs", READY_GLOBAL_CONST_TEXTURE_MEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyGlobalConstTexMemInstr = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("RGRMI", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"ReadyGlobalReadMemInstrs", READY_GLOBAL_READ_MEM_INSTRS, 2, 0, 1); //>=1
		        dRLSched->readyGlobalReadMemInstr = 0;
		        cmdPipeAttr = true;
		    }
		    else if (strcmp("WP", lToken) == 0)
		    {
		        addAttribute(lToken, (char*)"WhichPipeline", WHICH_PIPELINE, 4, 0, 1);
		
		        dRLSched->numReadyMemInstrs = 0;
		        dRLSched->numReadySfuInstrs = 0;
		        dRLSched->numReadySpInstrs = 0;
		        dRLSched->numReadyAluInstrs = 0;
		
		        rl_scheduler::gNumWarpsExecutingMemInstrGPU = 0;

				allocateGTCLongLatMemInstrCacheArr();
				allocateSFULongLatInstrCacheArr();
				allocateNumWarpsExecutingMemInstrArr();

		        cmdPipeAttr = true;
		    }
		}


		if (strstr(lToken, "ALPH") == lToken)
		{
			char* lTmp = lToken + strlen("ALPH");
			unsigned int lAlpha = atoi(lTmp);
			dRLSched->dCurrAlpha = (float)lAlpha/(float)100;
			dRLSched->dOrigAlpha = dRLSched->dCurrAlpha;
			if (sPrintFlag)
				printf("CurrAlpha=%f\n", dRLSched->dCurrAlpha);
		}
		else if (strstr(lToken, "XPL") == lToken)
		{
			char* lTmp = lToken + strlen("XPL");
			unsigned int lExploration = atoi(lTmp);
			dRLSched->dCurrExplorationPercent = lExploration;
			dRLSched->dOrigExplorationPercent = lExploration;
			if (sPrintFlag)
				printf("CurrExplorationPercent=%f\n", dRLSched->dCurrExplorationPercent);
		}
		else if (strstr(lToken, "GAM") == lToken)
		{
			char* lTmp = lToken + strlen("GAM");
			unsigned int lGamma = atoi(lTmp);
			if (lGamma == 100)
			{
				gGamma = 1.0;
			}
			else
			{
				gGamma = (float)lGamma/(float)10.0;
				while (gGamma > 1)
					gGamma = gGamma / 10.0;
			}
			if (sPrintFlag)
				printf("gGamma=%f\n", gGamma);
		}
		else if (strstr(lToken, "BET") == lToken)
		{
			char* lTmp = lToken + strlen("BET");
			unsigned int lBeta = atoi(lTmp);
			gBeta = (float)lBeta/(float)100;
			if (sPrintFlag)
				printf("gBeta=%f\n", gBeta);
		}
		else if (strstr(lToken, "PROJ") == lToken)
		{
			char* lTmp = lToken + strlen("PROJ");
			unsigned int lProj = atoi(lTmp);
			assert((lProj >= 0) && (lProj <= 31));
			gMaxProjVal = (1 << lProj);
			gMinProjVal = -gMaxProjVal;
			if (sPrintFlag)
				printf("gMaxProjVal=%d, gMinProjVal=%d\n", gMaxProjVal, gMinProjVal);
		}
		else if (strstr(lToken, "RWRD") == lToken)
		{
			char* lTmp = lToken + strlen("RWRD");
			unsigned int lReward = atoi(lTmp);
			gDiffReward = lReward;
			if (sPrintFlag)
				printf("gDiffReward=%u\n", gDiffReward);
		}
		else if (strstr(lToken, "PNLT") == lToken)
		{
			char* lTmp = lToken + strlen("PNLT");
			unsigned int lPenalty = atoi(lTmp);
			gDiffPenalty = lPenalty;
			if (sPrintFlag)
				printf("gDiffPenalty=%u\n", gDiffPenalty);
		}
		else if (strstr(lToken, "QTBL") == lToken)
		{
			char* lTmp = lToken + strlen("QTBL");
			unsigned int numBits = atoi(lTmp);
			gQtableSizeNumBits = numBits;
			assert(numBits < 32);
			unsigned int lQtableSize = 1 << numBits;
			gQtableSize = lQtableSize;
/*
			unsigned int lQtableSize = atoi(lTmp);
			gQtableSize = lQtableSize;
			unsigned int numBits = 0;
			while (lQtableSize > 0)
			{
				numBits++;
				lQtableSize = lQtableSize >> 1;
			}
			gQtableSizeNumBits = numBits;
*/
			if (sPrintFlag)
				printf("gQtableSize=%u, gQtableSizeNumBits=%u\n", gQtableSize, gQtableSizeNumBits);
		}
		else if (strstr(lToken, "SHDFR") == lToken)
		{
			char* lTmp = lToken + strlen("SHDFR");
			unsigned int lSchedFreq = atoi(lTmp);
			gSchedFreq = lSchedFreq;
			if (sPrintFlag)
				printf("gSchedFreq=%u\n", gSchedFreq);
		}
		else if (strstr(lToken, "NOPS") == lToken)
		{
			if (gUseMinAction == false)
			{
				char* lTmp = lToken + strlen("NOPS");
				unsigned int lMaxConsNoInstrs = atoi(lTmp);
				gMaxConsNoInstrs = lMaxConsNoInstrs;
				if (sPrintFlag)
					printf("gMaxConsNoInstrs=%u\n", gMaxConsNoInstrs);
			}
			else
			{
				gMaxConsNoInstrs = 0;
				printf("Using min action, so forcing gMaxConsNoInstrs = 0\n");
			}
		}
		else if (strstr(lToken, "QGTBL") == lToken)
		{
			char* lTmp = lToken + strlen("QGTBL");
			unsigned int lShare = atoi(lTmp);
			assert((lShare == 0) || (lShare == 1));
			if (lShare == 1)
				gShareQvalueTableForAllSMs = true;
			else 
				gShareQvalueTableForAllSMs = false;
			if (sPrintFlag)
				printf("gShareQvalueTableForAllSMs=%u\n", gShareQvalueTableForAllSMs);
		}
		else if (strstr(lToken, "TWOPH") == lToken)
		{
			char* lTmp = lToken + strlen("TWOPH");;
			unsigned int lTwoPhase = atoi(lTmp);
			assert((lTwoPhase == 0) || (lTwoPhase == 1));
			gTwoPhase = lTwoPhase;
			if (sPrintFlag)
				printf("gTwoPhase=%u\n", gTwoPhase);
		}
		else if (strstr(lToken, "UBKTS") == lToken)
		{
			char* lTmp = lToken + strlen("UBKTS");
			unsigned int lUniform = atoi(lTmp);
			assert((lUniform == 0) || (lUniform == 1));
			gUniformBuckets = lUniform;
			if (sPrintFlag)
				printf("gUniformBuckets=%u\n", gUniformBuckets);
		}
		else if (strstr(lToken, "BSHD") == lToken)
		{
			char* lTmp = lToken + strlen("BSHD");
			gBaseSched = atoi(lTmp);
			assert((gBaseSched >= 0) && (gBaseSched <= 4));
			if (sPrintFlag)
				printf("gBaseSched=%u\n", gBaseSched);
		}
		else if (strstr(lToken, "STNRY") == lToken)
		{
			char* lTmp = lToken + strlen("STNRY");
			unsigned int lStnryFlag = atoi(lTmp);
			gStnryFlag = lStnryFlag;
			if (sPrintFlag)
				printf("gStnryFlag=%u\n", gStnryFlag);
		}

		lToken = strtok(NULL, "_");
	}
	if (gRLSched && (gPrintResultDirExt == 1))
	{
		gPrintResultDirExt = 0;
		printf("ALPH = %f, XPL = %f, GAM = %f, RWRD=%u, PNLT=%u\n", dRLSched->dCurrAlpha, dRLSched->dCurrExplorationPercent/100.0, gGamma, gDiffReward, gDiffPenalty);
		//printf("RESULT_DIR_EXT=_%.2f_%.2f_%.2f_%u_%u_%u_%u_%u_%u_%u_%u\n", dRLSched->dCurrAlpha, dRLSched->dCurrExplorationPercent/100.0, gGamma, gDiffReward, gDiffPenalty, gQtableSize, gSchedFreq, gTwoPhase, gUniformBuckets, gMaxConsNoInstrs, gShareQvalueTableForAllSMs);
		printf("RESULT_DIR_EXT=_\n");
	}

	if (gUseCmdPipeTbTypeNumWarpsBypassAsAction)
	{
		if (dEngineNum == 0)
			dActionType = USE_CMD_PIPE_AS_ACTION;
		else if (dEngineNum == 1)
			dActionType = USE_TB_TYPE_AS_ACTION;
		else if (dEngineNum == 2)
			dActionType = USE_NUM_WARPS_AS_ACTION;
		else if (dEngineNum == 3)
			dActionType = USE_L1_BYPASS_AS_ACTION;
	}
	else if (gUseTbTypeAsAction)
		dActionType = USE_TB_TYPE_AS_ACTION;
	else if (gUseNAMaction)
		dActionType = USE_NAM_ACTION;
	else if (gUseNumOfWarpsAsAction && gUseBypassL1AsAction)
		dActionType = USE_NUM_WARPS_AND_L1_BYPASS_AS_ACTION;
	else if (gUseNumOfWarpsAsAction)
		dActionType = USE_NUM_WARPS_AS_ACTION;
	else if (gUseBypassL1AsAction)
		dActionType = USE_L1_BYPASS_AS_ACTION;
	else if (gUseWhichSchedAsAction)
		dActionType = USE_WHICH_SCHED_AS_ACTION;
	else if (gUseWhichWarpAsAction)
		dActionType = USE_WHICH_WARP_AS_ACTION;
	else if (gUseWhichWarpTypeAsAction)
		dActionType = USE_WHICH_WARP_TYPE_AS_ACTION;
	else if (gUseLrrGtoAsAction)
		dActionType = USE_LRR_GTO_AS_ACTION;
	else if (tbIdAttr)
    {
        if (cmdPipeAttr)
            dActionType = USE_TB_CMD_PIPE_AS_ACTION;
        else if (warpIdAttr)
            dActionType = USE_TB_WARP_ID_AS_ACTION;
        else
            dActionType = USE_TB_ID_AS_ACTION;
    }
    else if (warpIdAttr)
        dActionType = USE_WARP_ID_AS_ACTION;
    else if (cmdPipeAttr)
        dActionType = USE_CMD_PIPE_AS_ACTION;
    else if (tbTypeAttr)
        dActionType = USE_TB_TYPE_AS_ACTION;
	else if (lTBWSeen)
		dActionType = USE_TB_TYPE_AS_ACTION;
	else if (lNTFSeen)
		dActionType = USE_CMD_PIPE_AS_ACTION;
	else
        assert(0);

	setActionStrVec();
	
	//initialize stateVector used by function approximation code
	dCurrStateVector = new double[dAttributeVector.size()];	
    for (size_t i = 0; i < dAttributeVector.size(); i++)
		dCurrStateVector[i] = 0.0;

	if (gBaseSched == 4)
		gSetBaseSched();
}

void rlEngine::setActionStrVec()
{
	if (dActionType == USE_DUMMY_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_TB_ID_AS_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_WARP_ID_AS_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_CMD_PIPE_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("NO_INSTR"));
		dActionStrVec.push_back(std::string("SP_INSTR"));
		dActionStrVec.push_back(std::string("SFU_INSTR"));
		dActionStrVec.push_back(std::string("GMEM_INSTR"));
		dActionStrVec.push_back(std::string("STC_MEM_INSTR"));
	}
	else if (dActionType == USE_TB_CMD_PIPE_AS_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_TB_WARP_ID_AS_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_TB_TYPE_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("NO_TB"));
		dActionStrVec.push_back(std::string("FINISH_TB"));
		dActionStrVec.push_back(std::string("BARRIER_TB"));
		dActionStrVec.push_back(std::string("FASTEST_TB"));
		dActionStrVec.push_back(std::string("SLOWEST_TB"));
	}
	else if (dActionType == USE_NUM_WARPS_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_ONE_WARP"));
		dActionStrVec.push_back(std::string("SCHED_TWO_WARPS"));
		dActionStrVec.push_back(std::string("SCHED_FOUR_WARPS"));
		dActionStrVec.push_back(std::string("SCHED_EIGHT_WARPS"));
		dActionStrVec.push_back(std::string("SCHED_SIXTEEN_WARPS"));
		dActionStrVec.push_back(std::string("SCHED_ALL_WARPS"));
	}
	else if (dActionType == USE_L1_BYPASS_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("DO_NOT_BYPASS_L1"));
		dActionStrVec.push_back(std::string("BYPASS_L1"));
	}
	else if (dActionType == USE_NUM_WARPS_AND_L1_BYPASS_AS_ACTION)
	{
		assert(0);
	}
	else if (dActionType == USE_WHICH_SCHED_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_GTO_WARP"));
		dActionStrVec.push_back(std::string("SCHED_LRR_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YFB_WARP"));
		dActionStrVec.push_back(std::string("SCHED_MFS_WARP"));
		dActionStrVec.push_back(std::string("SCHED_FMS_WARP"));
	}
	else if (dActionType == USE_WHICH_WARP_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_GTO_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_WARP"));
		dActionStrVec.push_back(std::string("SCHED_NEXT_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_BARRIER_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_FINISH_WARP"));
		dActionStrVec.push_back(std::string("SCHED_OLDEST_SPLIT_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_SPLIT_WARP"));
		dActionStrVec.push_back(std::string("SCHED_OLDEST_LONG_LAT_MEM_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_LONG_LAT_MEM_WARP"));
	}
	else if (dActionType == USE_WHICH_WARP_TYPE_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_GTO_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_WARP"));
		dActionStrVec.push_back(std::string("SCHED_NEXT_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_BARRIER_WARP"));
		dActionStrVec.push_back(std::string("SCHED_YOUNGEST_FINISH_WARP"));
	}
	else if (dActionType == USE_NAM_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_NO_INSTR"));
		dActionStrVec.push_back(std::string("SCHED_ALU_INSTR"));
		dActionStrVec.push_back(std::string("SCHED_MEM_INSTR"));
	}
	else if (dActionType == USE_LRR_GTO_AS_ACTION)
	{
		dActionStrVec.push_back(std::string("SCHED_GTO_WARP"));
		dActionStrVec.push_back(std::string("SCHED_LRR_WARP"));
	}
	else
		assert(0);
}

void rlEngine::printQvalues()
{
	char fileName1[1024];
	char fileName2[1024];
	char fileName3[1024];

	if (dEngineNum == 0)
	{
		sprintf(fileName1, "qvalues.txt");
		sprintf(fileName2, "qvalueUpdateCnts.txt");
		if (rl_scheduler::gUseFeatureWeightFuncApprox)
			sprintf(fileName3, "weights.txt");
	}
	else
	{
		sprintf(fileName1, "qvalues%u.txt", dEngineNum+1);
		sprintf(fileName2, "qvalueUpdateCnts%u.txt", dEngineNum+1);
		if (rl_scheduler::gUseFeatureWeightFuncApprox)
			sprintf(fileName3, "weights%u.txt", dEngineNum+1);
	}

    FILE* qvaluesFile = fopen(fileName1, "w");
    FILE* qvalueUpdateCntsFile = fopen(fileName2, "w");
    FILE* weightsFile = 0;
	if (rl_scheduler::gUseFeatureWeightFuncApprox)
    	weightsFile = fopen(fileName3, "w");

	printQvalues(qvaluesFile, qvalueUpdateCntsFile, weightsFile);

	fclose(qvaluesFile);
	fclose(qvalueUpdateCntsFile);
	if (rl_scheduler::gUseFeatureWeightFuncApprox)
		fclose(weightsFile);
}

void rl_scheduler::printQvalues()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->printQvalues();
}

void valueMap::getStats(unsigned int& xNumQvalueEntriesVisited, float& xTotalQvalue)
{
	for (std::map<unsigned long long, float>::iterator lIter = dStateActionValueMap.begin();
		 lIter != dStateActionValueMap.end();
		 lIter++)
	{
		xTotalQvalue += lIter->second;
		xNumQvalueEntriesVisited++;
	}
}

void valueUpdateMap::getStats(unsigned int& xTotalQvalueUpdates)
{
	for (std::map<unsigned long long, unsigned int>::iterator lIter = dMap.begin();
		 lIter != dMap.end();
		 lIter++)
	{
		xTotalQvalueUpdates += lIter->second;
	}
}

std::string rlEngine::getActionStr(unsigned int actionVal)
{
	if (actionVal >= dActionStrVec.size())
		return dActionStrVec[0];
	else
		return dActionStrVec[actionVal];
	//return dActionStrVec.at(actionVal);
}

std::string rlEngine::getStateNameValStr(unsigned int stateVal)
{
	unsigned int lStateVal = stateVal;
	char stateNameValStr[1024];
	stateNameValStr[0] = '\0';
    for (int i = dAttributeVector.size() - 1; i >= 0; i--)
    {
        rl_attribute& attr = dAttributeVector[i];
		assert(attr.dPlaceValue != 0xdeaddead);
		unsigned int attrVal = lStateVal / attr.dPlaceValue;
		char tmp[15];
		sprintf(tmp, " %u ", attrVal);
		strcat(stateNameValStr, attr.attrShortName);
		strcat(stateNameValStr, tmp);
		lStateVal = lStateVal % attr.dPlaceValue;
    }
	return std::string(stateNameValStr);
}

std::string rlEngine::getStateValStr(unsigned int stateVal)
{
	unsigned int lStateVal = stateVal;
	char stateValStr[1024];
	stateValStr[0] = '\0';
    for (int i = dAttributeVector.size() - 1; i >= 0; i--)
    {
        rl_attribute& attr = dAttributeVector[i];
		assert(attr.dPlaceValue != 0xdeaddead);
		unsigned int attrVal = lStateVal / attr.dPlaceValue;
		char tmp[15];
		sprintf(tmp, " %u", attrVal);
		strcat(stateValStr, tmp);
		lStateVal = lStateVal % attr.dPlaceValue;
    }
	return std::string(stateValStr);
}

std::string rlEngine::getStateNameStr()
{
	char stateNameStr[1024];
	stateNameStr[0] = '\0';
    for (int i = dAttributeVector.size() - 1; i >= 0; i--)
    {
        rl_attribute& attr = dAttributeVector[i];
		strcat(stateNameStr, " ");
		strcat(stateNameStr, attr.attrShortName);
    }
	return std::string(stateNameStr);
}

unsigned int gIterNum = 1;

void rlEngine::printQvalues(FILE* qvaluesFile, FILE* qvalueUpdateCntsFile, FILE* weightsFile)
{
    unsigned int smId = dRLSched->m_shader->get_sid();
	unsigned int schedId = dRLSched->m_id;
    printf("RL: QvalueUpdateCounts for SM %u Sched %u\n", smId, schedId);

    unsigned int totalQvalueUpdates = 0;

	fprintf(qvaluesFile, "%s\n", dAttrString.c_str());
	fprintf(qvaluesFile, "%f\n", dRLSched->dCurrAlpha);
	fprintf(qvaluesFile, "%f\n", dRLSched->dCurrExplorationPercent);
	fprintf(qvaluesFile, "%u\n", gIterNum);

	unsigned int lNumQvalueEntriesVisited = 0;
    float totalQvalue = 0.0;

	bool lLastSchedValues = true;
	if (lLastSchedValues)
	{
		dValues->getStats(lNumQvalueEntriesVisited, totalQvalue);
		dValueUpdates->getStats(totalQvalueUpdates);

		printf("state action update table\n");
		printf("=========================\n");
    	for (unsigned long long i = 0; i < dStateActionValueArraySize; i++)
    	{
			float lQvalue = mGetQvalue(i);
			unsigned int lQvalueUpdates = mGetQvalueUpdate(i);
			if (lQvalueUpdates != 0)
			{
        		fprintf(qvaluesFile, "%llu %e\n", i, lQvalue);
        		fprintf(qvalueUpdateCntsFile, "%llu %u\n", i, lQvalueUpdates);
			}
			/*
			if (lQvalueUpdates != 0)
			{
				unsigned int stateVal = i / dNumActions;
				std::string stateStr = getStateNameValStr(stateVal);
				unsigned int actionVal = i % dNumActions;
				std::string actionStr = getActionStr(actionVal);
				printf("state %u (%s), action %u (%s), updates %u\n", stateVal, stateStr.c_str(), actionVal, actionStr.c_str(), lQvalueUpdates);
				if (gStateActionUpdateCntMap.find(stateVal) == gStateActionUpdateCntMap.end())
				{
					std::vector<unsigned int> actionUpdateVec;
					for (unsigned int i = 0; i < dNumActions; i++)
						actionUpdateVec.push_back(0);
					
					actionUpdateVec[actionVal] = lQvalueUpdates;
					gStateActionUpdateCntMap[stateVal] = actionUpdateVec;
				}
				else
				{
					std::vector<unsigned int>& actionUpdateVec = gStateActionUpdateCntMap[stateVal];
					actionUpdateVec[actionVal] = lQvalueUpdates;
				}
			}
			*/
    	}

		if (rl_scheduler::gUseFeatureWeightFuncApprox)
		{
			for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
			{
				float lWt = dStateActionWeightArray[i];
				fprintf(weightsFile, "%u %e\n", i, lWt);
			}
		}

		if (gStateActionUpdateCntMapEnabled)
		{
			printf("states actions csv\n");
			printf("=========================\n");
			FILE* sac_csv_fp = fopen("sac.csv", "w");
			fprintf(sac_csv_fp, "state");
			std::string stateNameStr = getStateNameStr();
			fprintf(sac_csv_fp, "%s", stateNameStr.c_str());
			for (unsigned int i = 0; i < dNumActions; i++)
				fprintf(sac_csv_fp, " %s", getActionStr(i).c_str());
			fprintf(sac_csv_fp, "\n");
	
			for (std::map<unsigned int, std::vector<unsigned int> >::iterator iter = gStateActionUpdateCntMap.begin();
			 	iter != gStateActionUpdateCntMap.end();
			 	iter++)
			{
				unsigned int stateVal = iter->first;
				fprintf(sac_csv_fp, "%u", stateVal);
				std::string stateValStr = getStateValStr(stateVal);
				fprintf(sac_csv_fp, "%s", stateValStr.c_str());
				std::vector<unsigned int>& actionUpdateVec = iter->second;
				for (unsigned int i = 0; i < actionUpdateVec.size(); i++)
				{
					unsigned int avg = actionUpdateVec[i] / (gNumSMs * NUM_SCHED_PER_SM);
					unsigned int rem = actionUpdateVec[i] % (gNumSMs * NUM_SCHED_PER_SM);
					if (rem)
						avg++;
					//fprintf(sac_csv_fp, " %u", actionUpdateVec[i]);
					//fprintf(sac_csv_fp, " %u(%u)", avg, actionUpdateVec[i]);
					fprintf(sac_csv_fp, " %u", avg);
				}
				fprintf(sac_csv_fp, "\n");
			}
			gStateActionUpdateCntMap.clear();
			fclose(sac_csv_fp);
		}
	}
	printf("Total q value array size = %llu, visited entries = %u\n", dStateActionValueArraySize, lNumQvalueEntriesVisited);

    printf("RL: Total dValueUpdates %u, total of all updated dValues = %.2f for SM %u Sched %u\n", totalQvalueUpdates, totalQvalue, smId, schedId);

	if (dSarsaAgent)
	{
		printf("begin weights and traces from CMAC:\n");
		//((CMAC*)(dSarsaAgent->FA))->print();
		//dCMAC->print();
		printf("end weights and traces from CMAC:\n");
	}
}

void rl_scheduler::clear()
{
	//printf("rl_scheduler::clear\n");
    //this function is called at the end of a kernel, so if there are multiple kernels
    //pass the existing dValues so that a new dValues table is not created and the existing
    //one is initialized to default values

	if (rl_scheduler::gUsePrevQvalues == false)
	{
    	initQvalues();
    	initQvalueUpdates();
	}

    initCurrStateAndAction();

    if (rl_scheduler::gNumWarpsExecutingMemInstr)
        rl_scheduler::gNumWarpsExecutingMemInstr[m_shader->get_sid()] = 0;
    if (rl_scheduler::gNumReadyMemInstrsWithSameTB)
        rl_scheduler::gNumReadyMemInstrsWithSameTB[m_shader->get_sid()] = 0;
    if (rl_scheduler::gLastMemInstrTB)
        rl_scheduler::gLastMemInstrTB[m_shader->get_sid()] = 0;
    if (rl_scheduler::gNumReadyMemInstrsWithSamePC)
        rl_scheduler::gNumReadyMemInstrsWithSamePC[m_shader->get_sid()] = 0;
    if (rl_scheduler::gLastMemInstrPC)
        rl_scheduler::gLastMemInstrPC[m_shader->get_sid()] = 0;
    if (rl_scheduler::gNumWarpsWaitingAtBarrier)
        rl_scheduler::gNumWarpsWaitingAtBarrier[m_shader->get_sid()] = 0;
    if (rl_scheduler::gNumWarpsFinished)
        rl_scheduler::gNumWarpsFinished[m_shader->get_sid()] = 0;

	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->mClear();

	dConsecutiveNoInstrSchedCnt = 0;
	dFirstWarpIssued = false;

	scheduler_unit::clear();
}

void rl_scheduler::initQvalues()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->initQvalues();
}

void rlEngine::initQvalues()
{
	unsigned int smId = dRLSched->m_shader->get_sid();
	uint schedId = dRLSched->m_id;

	bool lDefaultInit = true;
    if (rl_scheduler::gUsePrevQvalues)
	{
    	FILE* fp = 0;
    	FILE* fp2 = 0;
    	FILE* fp3 = 0;
		if (dEngineNum == 1)
		{
    		fp = fopen("qvalues2.txt", "r");
    		fp2 = fopen("qvalueUpdateCnts2.txt", "r");
			if (rl_scheduler::gUseFeatureWeightFuncApprox)
    			fp3 = fopen("weights2.txt", "r");
		}
		else
		{
    		fp = fopen("qvalues.txt", "r");
    		fp2 = fopen("qvalueUpdateCnts.txt", "r");
			if (rl_scheduler::gUseFeatureWeightFuncApprox)
    			fp3 = fopen("weights.txt", "r");
		}
		if ((fp == NULL) || (fp2 == NULL))
		{
    		dValues->mClear();
    		dValueUpdates->mClear();
		}
		else
		{
    		if (fp != NULL)
    		{
				float lLastAlpha = 0.0;
				unsigned int lLastEpsilon = 0;
	
				lDefaultInit = false;
        		unsigned int i = 0;
				unsigned int cnt = 0;
        		while (!feof(fp))
        		{
            		char line[1024];
            		char* ptr = fgets(line, 1024, fp);
            		if (ptr == NULL)
                		break;
					if (i == 0)
					{
						char* lNewLine = strstr(ptr, "\n");
						assert(lNewLine);
						*lNewLine = '\0';
						if (strcmp(dAttrString.c_str(), ptr) != 0)
						{
							lDefaultInit = true;
							break;
						}
					}
					else if (i == 1)
					{
						lLastAlpha = atof(ptr);
					}
					else if (i == 2)
					{
						lLastEpsilon = atoi(ptr);
					}
					else if ((i == 3) && (gUseWhichSchedAsAction == false))
					{
						unsigned int lLastIterNum = atoi(ptr);
						gIterNum = lLastIterNum + 1;
						float newExpl = 0.0;
						float newAlpha = 0.0;
						if (gIterNum > 100)
							newExpl = 1.0;
						else
						{
							newExpl = (dRLSched->dOrigExplorationPercent * (101 - gIterNum)) / 100.0;
							newAlpha = (dRLSched->dOrigAlpha * (101 - gIterNum)) / 100.0;
							if (newExpl < 1.0)
								newExpl = 1.0;
							if (newAlpha < 0.01)
								newAlpha = 0.01;
							printf("%u th iteration so changing exploration from %f to %f, alpha from %f to %f\n", gIterNum,  dRLSched->dOrigExplorationPercent, newExpl, dRLSched->dOrigAlpha, newAlpha);
						}
						dRLSched->dOrigExplorationPercent = newExpl;
						dRLSched->dOrigAlpha = newAlpha;
					}
					else
					{
            			//double qValue = atof(ptr);
            			float qValue;
						sscanf(ptr, "%u %e\n", &cnt, &qValue);
            			if (cnt >= dStateActionValueArraySize)
            			{
                			printf("2. incompatible qvalues.txt file, sm %u sched %u\n", smId, schedId);
							lDefaultInit = true;
                			break;
            			}
						mSetQvalue(cnt, qValue);
						//cnt++;
					}
            		i++;
        		}
				if (lDefaultInit == false)
				{
					printf("Initialized q value table using prev q values\n");
				}
				else
				{
    				dValues->mClear();
    				dValueUpdates->mClear();
				}
	
				fclose(fp);
    		}
			if (fp2 != NULL)
			{
				lDefaultInit = false;
        		unsigned int i = 0;
        		while (!feof(fp2))
        		{
            		char line[1024];
            		char* ptr = fgets(line, 1024, fp2);
            		if (ptr == NULL)
                		break;
					unsigned int cnt = 0;
            		unsigned int updateCnt;
					sscanf(ptr, "%u %u\n", &cnt, &updateCnt);
            		//unsigned int updateCnt = atoi(ptr);
            		if (cnt >= dStateActionValueArraySize)
            		{
                		printf("2. incompatible qvalueUpdateCnts2.txt file, sm %u sched %u\n", smId, schedId);
						lDefaultInit = true;
                		break;
            		}
					mSetQvalueUpdateCnt(cnt, updateCnt);
            		i++;
        		}
				if (lDefaultInit == false)
				{
					printf("Initialized q value update count table using prev q value update count table\n");
				}
				else
				{
    				dValues->mClear();
    				dValueUpdates->mClear();
				}
	
				fclose(fp2);
			}
			if (fp3 != NULL)
			{
				lDefaultInit = false;
        		unsigned int i = 0;
        		while (!feof(fp3))
        		{
            		char line[1024];
            		char* ptr = fgets(line, 1024, fp3);
            		if (ptr == NULL)
                		break;
					unsigned int cnt = 0;
            		float lWt;
					sscanf(ptr, "%u %e\n", &cnt, &lWt);
            		if (cnt >= (dAttributeVector.size() * dNumActions))
            		{
                		printf("3. incompatible weights.txt file, sm %u sched %u\n", smId, schedId);
						lDefaultInit = true;
                		break;
            		}
					dStateActionWeightArray[cnt] = lWt;
            		i++;
        		}
				if (lDefaultInit == false)
				{
					printf("Initialized weights array using prev weights array\n");
				}
				else
				{
            		for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
						dStateActionWeightArray[i] = 0;
				}
				fclose(fp3);
			}
		}
	}
}

void rl_scheduler::initQvalueUpdates()
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->initQvalueUpdates();
}

void rlEngine::initQvalueUpdates()
{
}

valueMap* rlEngine::allocateQvalues(valueMap* xQvalueTable)
{
    dStateActionValueArraySize = getQvalueMatrixSize();

	if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
	{
		dPrevQForFA = 0.0;
		if (dRLSched->m_id == 0)
		{
			dStateActionFeatureValueArray = new float[dAttributeVector.size() * dNumActions];
			dStateActionWeightArray = new float[dAttributeVector.size() * dNumActions];
			for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
			{
				dStateActionFeatureValueArray[i] = 0.0;
				dStateActionWeightArray[i] = 0.0;
				std::vector<float> f;
				std::map<unsigned int, std::vector<float> >& lWeightsVecMap = gWeightsVecMapVec[dEngineNum];
				lWeightsVecMap[i] = f;
			}
			if (gNewActorCriticMethod)
			{
				dPrev_vXphi = 0.0;
				dStateActionWeightArray2 = new float[dAttributeVector.size() * dNumActions];
				for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
					dStateActionWeightArray2[i] = 0.0;
			}
	
			if (gActorCriticMethod)
			{
				dStateFeatureValueArray = new float[dAttributeVector.size()];
				dStateWeightArray = new float[dAttributeVector.size()];
				dPrevSForFA = 0.0;
				dPrevHForFA = 0.0;
				for (unsigned int i = 0; i < dAttributeVector.size(); i++)
				{
					dStateFeatureValueArray[i] = 0.0;
					dStateWeightArray[i] = 0.0;
					std::vector<float> f;
				}
			}
		}
		else
		{
			shader_core_ctx* shaderCore = dRLSched->m_shader;
			assert(shaderCore->schedulers.size() == 2);
			rl_scheduler* sched0 = (rl_scheduler*) shaderCore->schedulers[0];

			dStateActionFeatureValueArray = sched0->dRLEngines[0]->dStateActionFeatureValueArray;
			dStateActionWeightArray = sched0->dRLEngines[0]->dStateActionWeightArray;
			if (gNewActorCriticMethod)
				dStateActionWeightArray2 = sched0->dRLEngines[0]->dStateActionWeightArray2;
			if (gActorCriticMethod)
			{
				dStateFeatureValueArray = sched0->dRLEngines[0]->dStateFeatureValueArray;
				dStateWeightArray = sched0->dRLEngines[0]->dStateWeightArray;
			}
		}
	}

	if (dStateActionValueArraySize > gQtableSize)
	{
		printf("Reducing q value array size from %llu to %u\n", dStateActionValueArraySize, gQtableSize);
		dStateActionValueArraySize = gQtableSize;
	}

	if (xQvalueTable == 0)
		dValues = new valueMap;
	else
		dValues = xQvalueTable;
	return dValues;
}

valueUpdateMap* rlEngine::allocateQvalueUpdates(valueUpdateMap* xQvalueUpdateTable)
{
    dStateActionValueArraySize = getQvalueMatrixSize();
	if (dStateActionValueArraySize > gQtableSize)
	{
		printf("Reducing q value array size from %llu to %u\n", dStateActionValueArraySize, gQtableSize);
		dStateActionValueArraySize = gQtableSize;
	}

    if (xQvalueUpdateTable == 0)
		dValueUpdates = new valueUpdateMap;
	else
		dValueUpdates = xQvalueUpdateTable;

	return dValueUpdates;
}

rl_scheduler::rl_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                               Scoreboard* scoreboard, simt_stack** simt,
                               std::vector<shd_warp_t>* warp,
                               register_set* sp_out,
                               register_set* sfu_out,
                               register_set* mem_out,
                               int id,
							   unsigned int xNumRLEngines)
    : scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id )
{
	dNumRLEngines = 0;
	dRLEngines[0] = new rlEngine(this, dNumRLEngines);
	dNumRLEngines++;
	if (xNumRLEngines == 2)
	{
		dRLEngines[1] = new rlEngine(this, dNumRLEngines);
		dNumRLEngines++;
	}
	else if (xNumRLEngines == 4)
	{
		dRLEngines[1] = new rlEngine(this, dNumRLEngines);
		dNumRLEngines++;
		dRLEngines[2] = new rlEngine(this, dNumRLEngines);
		dNumRLEngines++;
		dRLEngines[3] = new rlEngine(this, dNumRLEngines);
		dNumRLEngines++;
	}
	else
	{
		dRLEngines[1] = 0;
		dRLEngines[2] = 0;
		dRLEngines[3] = 0;
	}
	dConsecutiveNoInstrSchedCnt = 0;
	dFirstWarpIssued = false;
	dCurrLrrGtoAction = GTO_SCHED;

	dOrigAlpha = ALPHA;
	dCurrAlpha = ALPHA;

	dOrigExplorationPercent = EXPLORATION_PERCENT;
	dCurrExplorationPercent = EXPLORATION_PERCENT;

	dFirstTime = true;
	for (unsigned int i = 0; i < 50; i++)
		gPossiblePrimaryActionCntMap[i] = 0;
	dFirstMaxQAction = 0xdeaddead;
	dNumOfSWLwarps = 32;
}
unsigned int gNumReadyMemInstrWarps = 0;
unsigned int gNumReadyAluInstrWarps = 0;
unsigned int gNumWarpsWaitingForData = 0;

void checkSplitWarp(std::map<unsigned int, unsigned int>& splitWarpDynamicIdMap, unsigned int dyn_warp_id, 
					unsigned int activeMaskCount, shd_warp_t* warpPtr, unsigned int& numSplitWarps)
{
	std::map<unsigned int, unsigned int>::iterator iter1 = splitWarpDynamicIdMap.find(dyn_warp_id);
	if (iter1 == splitWarpDynamicIdMap.end())
		splitWarpDynamicIdMap[dyn_warp_id] = activeMaskCount;
	else if (iter1->second != activeMaskCount)
	{
		warpPtr->mSetSplitWarp();
		numSplitWarps++;
	}
	else
		warpPtr->mResetSplitWarp();
}

void rl_scheduler::initRLAttributeArrays(unsigned int smId)
{
    //rl_scheduler::gGTCLongLatMemInstrReady = false;
    rl_scheduler::gSFULongLatInstrReady = false;
    numReadyGlobalMemInstrs = 0;
    numReadySharedMemInstrs = 0;
	numReadySharedTexConstMemInstrs = 0;
    numReadyConstantMemInstrs = 0;
    numReadyTextureMemInstrs = 0;
    numReadyReadMemInstrs = 0;
    numReadyWriteMemInstrs = 0;
	numFutureMemInstrs = 0;
	numFutureSfuInstrs = 0;
    numReadyMemInstrs = 0;
    numReadySfuInstrs = 0;
    numReadySpInstrs = 0;
	numSplitWarps = 0;
    numReadyAluInstrs = 0;
    numReadyInstrs = 0;
	numSchedulableWarps = 0;
	numWaitingInstrs = 0; //waiting for data
	numPipeStalls = 0;
	numMemPipeStalls = 0;
	numSfuPipeStalls = 0;
	numSpPipeStalls = 0;
	numIdleWarps = 0;

    if (rl_scheduler::gNumReadyMemInstrsWithSameTB)
        rl_scheduler::gNumReadyMemInstrsWithSameTB[smId] = 0;
    if (rl_scheduler::gNumReadyMemInstrsWithSamePC)
        rl_scheduler::gNumReadyMemInstrsWithSamePC[smId] = 0;
    if (rl_scheduler::gNumWarpsWaitingAtBarrier)
        rl_scheduler::gNumWarpsWaitingAtBarrier[smId] = 0;
    if (rl_scheduler::gNumWarpsFinished)
        rl_scheduler::gNumWarpsFinished[smId] = 0;

    readySpInstrs = 0;
    readySfuInstrs = 0;
    readyMemInstrs = 0;
    readyGlobalMemInstrs = 0;
    readyLongLatMemInstrs = 0;
    readySharedMemInstrs = 0;
    readyConstMemInstrs = 0;
    readyTexMemInstrs = 0;
    readyGlobalConstTexReadMemInstr = 0;
	readySharedTexConstMemInstr = 0;
    readyGlobalConstTexMemInstr = 0;
    readyGlobalReadMemInstr = 0;

    rl_scheduler::gNumMemSchedQsLoaded = 0;
    rl_scheduler::gNumReqsInMemSchedQs = 0;
    if (gNumReqsInMemSchedArray)
    {
        for (unsigned int i = 0; i < 6; i++)
        {
            rl_scheduler::gNumReqsInMemSchedQs += gNumReqsInMemSchedArray[i];
            if (gNumReqsInMemSchedArray[i] > 12)
                rl_scheduler::gNumMemSchedQsLoaded++;
        }
    }

    std::map<unsigned int, unsigned int>& splitWarpDynamicIdMap = gSplitWarpDynamicIdMapVec.at(smId);

    for (std::vector<shd_warp_t*>::const_iterator iter = m_supervised_warps.begin(); 
         iter != m_supervised_warps.end(); 
         iter++) 
    {
        // Don't consider warps that are not yet valid
        if ((*iter) == NULL)
               continue;
        if ((*iter)->done_exit())
        {
            if (rl_scheduler::gNumWarpsFinished)
                rl_scheduler::gNumWarpsFinished[smId]++;
            continue;
        }

		shd_warp_t* warpPtr = (*iter);
        unsigned warp_id = warpPtr->get_warp_id();
        unsigned int tbId = warpPtr->get_cta_id();
		unsigned int dyn_warp_id = warpPtr->get_dynamic_warp_id();

        if (!warp(warp_id).ibuffer_empty())
        {
            const warp_inst_t* pI1 = warp(warp_id).ibuffer_next_inst();

            if (m_shader->warp_waiting_at_barrier(warp_id))
            {
                if (rl_scheduler::gNumWarpsWaitingAtBarrier)
                    rl_scheduler::gNumWarpsWaitingAtBarrier[smId]++;
                
				numIdleWarps++;
                continue;
            }
            if(pI1) 
            {
                unsigned pc, rpc;

                m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc, &rpc);

                if( pc == pI1->pc ) 
                {
					warpPtr->mResetLongLatMemInstr();
                    if (!m_scoreboard->checkCollision(warp_id, pI1)) 
                    {
                        numReadyInstrs++;
                		const active_mask_t &active_mask = m_simt_stack[warp_id]->get_active_mask();
                        unsigned int activeMaskCount = active_mask.count();
                        if ((pI1->op == LOAD_OP) || (pI1->op == STORE_OP) || (pI1->op == MEMORY_BARRIER_OP)) 
                        {
                            if(m_mem_out->has_free()) 
							{
                                gReadyTBIdSet.insert(tbId);
								numSchedulableWarps++;
							}
							else
							{
								numMemPipeStalls++;
								numPipeStalls++;
							}
                            numReadyMemInstrs++;
							readyMemInstrs = 1;

							checkSplitWarp(splitWarpDynamicIdMap, dyn_warp_id, activeMaskCount, warpPtr, numSplitWarps);

                            if (rl_scheduler::gGTCLongLatMemInstrCache)
                            {
                                for (unsigned int i = 0; i < GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE; i++)
                                {
                                    if (pc == rl_scheduler::gGTCLongLatMemInstrCache[(smId * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE) + i])
                                    {
                                        //rl_scheduler::gGTCLongLatMemInstrReady = true;
										readyLongLatMemInstrs = true;
										warpPtr->mSetLongLatMemInstr();
                                        break;
                                    }
                                }
                            }

                            if (pI1->space.get_type() == global_space)
							{
                                numReadyGlobalMemInstrs++;
								readyGlobalMemInstrs = 1;
							}

                            if (pI1->space.get_type() == shared_space)
							{
								numReadySharedTexConstMemInstrs++;
                                numReadySharedMemInstrs++;
								readySharedMemInstrs = 1;
								readySharedTexConstMemInstr = 1;
							}

                            if (pI1->space.get_type() == const_space)
							{
								numReadySharedTexConstMemInstrs++;
                                numReadyConstantMemInstrs++;
								readyConstMemInstrs = 1;
								readySharedTexConstMemInstr = 1;
							}

                            if (pI1->space.get_type() == tex_space)
							{
								numReadySharedTexConstMemInstrs++;
                                numReadyTextureMemInstrs++;
								readyTexMemInstrs = 1;
								readySharedTexConstMemInstr = 1;
							}

                            if (pI1->op == LOAD_OP)
                                numReadyReadMemInstrs++;

                            if (pI1->op == STORE_OP)
                                numReadyWriteMemInstrs++;

                            if (rl_scheduler::gLastMemInstrTB && rl_scheduler::gNumReadyMemInstrsWithSameTB)
                                if ((*iter)->get_cta_id() == rl_scheduler::gLastMemInstrTB[smId])
                                    rl_scheduler::gNumReadyMemInstrsWithSameTB[smId]++;

                            if (rl_scheduler::gLastMemInstrPC && rl_scheduler::gNumReadyMemInstrsWithSamePC)
                                if (pI1->pc == rl_scheduler::gLastMemInstrPC[smId])
                                    rl_scheduler::gNumReadyMemInstrsWithSamePC[smId]++;

                            if ((pI1->space.get_type() == global_space) || 
                                (pI1->space.get_type() == const_space) ||
                                (pI1->space.get_type() == tex_space))
                            {
                                readyGlobalConstTexMemInstr = 1;
                            }

                            if ((pI1->op == LOAD_OP) || (pI1->op == MEMORY_BARRIER_OP))
                            {
                                if ((pI1->space.get_type() == global_space) || 
                                    (pI1->space.get_type() == const_space) ||
                                    (pI1->space.get_type() == tex_space))
                                {
                                    readyGlobalConstTexReadMemInstr = 1;
                                }
                                if (pI1->space.get_type() == global_space)
                                    readyGlobalReadMemInstr = 1;
                            }
                        }
                        else 
                        {
                            bool sfu_pipe_avail = m_sfu_out->has_free();
                            bool sp_pipe_avail = m_sp_out->has_free();

							if (pI1->op == SFU_OP)
							{
                            	numReadyAluInstrs++;
                            	numReadySfuInstrs++;
                            	readySfuInstrs = 1;

								checkSplitWarp(splitWarpDynamicIdMap, dyn_warp_id, activeMaskCount, warpPtr, numSplitWarps);

								if (sfu_pipe_avail)
								{
									gReadyTBIdSet.insert(tbId);
									numSchedulableWarps++;
								}
								else
								{
									numSfuPipeStalls++;
									numPipeStalls++;
								}
							}
							else if (pI1->op == ALU_SFU_OP)
							{
                            	numReadyAluInstrs++;
                            	numReadySfuInstrs++;
                            	readySfuInstrs = 1;

								checkSplitWarp(splitWarpDynamicIdMap, dyn_warp_id, activeMaskCount, warpPtr, numSplitWarps);

                            	numReadySpInstrs++;
                            	readySpInstrs = 1;

								if (sfu_pipe_avail || sp_pipe_avail)
								{
									gReadyTBIdSet.insert(tbId);
									numSchedulableWarps++;
								}
								else
								{
									numSfuPipeStalls++;
									numSpPipeStalls++;
									numPipeStalls++;
								}
							}
							else
							{
								numReadyAluInstrs++;
                            	numReadySpInstrs++;
                            	readySpInstrs = 1;

								checkSplitWarp(splitWarpDynamicIdMap, dyn_warp_id, activeMaskCount, warpPtr, numSplitWarps);

								if (sp_pipe_avail)
								{
									gReadyTBIdSet.insert(tbId);
									numSchedulableWarps++;
								}
								else
								{
									numSpPipeStalls++;
									numPipeStalls++;
								}
							}

                            if (rl_scheduler::gSFULongLatInstrCache)
                            {
                                for (unsigned int i = 0; i < SFU_LONG_LAT_INSTR_CACHE_SIZE; i++)
                                {
                                    if (pc == rl_scheduler::gSFULongLatInstrCache[(smId * SFU_LONG_LAT_INSTR_CACHE_SIZE) + i])
                                    {
                                        rl_scheduler::gSFULongLatInstrReady = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
					else
						numWaitingInstrs++;
                }
				else
					numIdleWarps++;
            } 
			else
				numIdleWarps++;
            if (warp(warp_id).ibuffer_next_to_next_inst())
			{
            	const warp_inst_t* pI2 = warp(warp_id).ibuffer_next_to_next_inst();
				if (pI2)
				{
                	if ((pI2->op == LOAD_OP) || (pI2->op == STORE_OP) || (pI2->op == MEMORY_BARRIER_OP)) 
						numFutureMemInstrs++;
                    else if ((pI2->op == SFU_OP) || (pI2->op == ALU_SFU_OP))
						numFutureSfuInstrs++;
				}
			}
        }
		else
			numIdleWarps++;
	}
	{
		gNumReadyMemInstrWarps += numReadyMemInstrs;
		gNumReadyAluInstrWarps += (numReadySfuInstrs + numReadySpInstrs);
		gNumWarpsWaitingForData += numWaitingInstrs;
	}
}

unsigned int gPrimaryFirstCnt = 0;
unsigned int gSecondaryFirstCnt = 0;

unsigned int gLastPrintCycle = 0;
int gTotalGPUWideRewardPrev = 0;

unsigned int rlEngine::mGetWhichSchedAction(bool exploration, long int randVal)
{
	unsigned int action = 0xdeaddead;
/*
	if (exploration || dFirstTime)
	{
		uint maxNumActions = MAX_ACTIONS_OF_TYPE_WHICH_SCHED;

		bool done = false;
		while (done == false)
		{
			action = randVal % maxNumActions;
			if (action == SCHED_YFB_WARP)
			{
				if (gFinishTbIdSet.empty() && gBarrierTbIdSet.empty())
				{
        			randVal = random();
					continue;
				}
			}
			done = true;
		}

		return action;
	}
*/
	unsigned int lPrimaryMaxQAction = 0xdeaddead;
	float lPrimaryMaxQvalue = 0.0;
	for (unsigned int i = 0; i < MAX_ACTIONS_OF_TYPE_WHICH_SCHED; i++)
	{
		if (i == SCHED_YFB_WARP)
		{
			if (gFinishTbIdSet.empty() && gBarrierTbIdSet.empty())
				continue;
		}
		unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
     	unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
     	float lQvalue = mGetQvalue(lIndex);
		if (lPrimaryMaxQAction == 0xdeaddead)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
		else if (lQvalue > lPrimaryMaxQvalue)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
	}
	action = lPrimaryMaxQAction;
	return action;
}

unsigned int rlEngine::mGetWhichWarpAction(bool exploration, long int randVal)
{
	unsigned int action = 0xdeaddead;
	//if (exploration || dFirstTime)
	if (exploration)
	{
		uint maxNumActions = MAX_ACTIONS_OF_TYPE_WHICH_WARP;

		bool done = false;
		while (done == false)
		{
			action = randVal % maxNumActions;
	
			if (action == SCHED_YOUNGEST_BARRIER_WARP)
			{
				if (gBarrierTbIdSet.empty())
				{
        			randVal = random();
					continue;
				}
			}
			else if (action == SCHED_YOUNGEST_FINISH_WARP)
			{
				if (gFinishTbIdSet.empty())
				{
        			randVal = random();
					continue;
				}
			}
			else if ((action == SCHED_OLDEST_SPLIT_WARP) || (action == SCHED_YOUNGEST_SPLIT_WARP))
			{
				if (dRLSched->numSplitWarps == 0)
				{
        			randVal = random();
					continue;
				}
			}
			else if ((action == SCHED_OLDEST_LONG_LAT_MEM_WARP) || (action == SCHED_YOUNGEST_LONG_LAT_MEM_WARP))
			{
				if (dRLSched->readyLongLatMemInstrs == false)
				{
        			randVal = random();
					continue;
				}
			}
			done = true;
		}
		return action;
	}

	unsigned int lPrimaryMaxQAction = 0xdeaddead;
	float lPrimaryMaxQvalue = 0.0;
    //unsigned int smId = dRLSched->m_shader->get_sid();
	//unsigned int schedId = dRLSched->m_id;
	//if (smId == 0)
		//printf("%llu: sched %u q values: ", gpu_sim_cycle,  schedId);
	for (unsigned int i = 0; i < MAX_ACTIONS_OF_TYPE_WHICH_WARP; i++)
	{
		if (i == SCHED_YOUNGEST_BARRIER_WARP)
		{
			if (gBarrierTbIdSet.empty())
				continue;
		}
		else if (i == SCHED_YOUNGEST_FINISH_WARP)
		{
			if (gFinishTbIdSet.empty())
				continue;
		}
		else if ((i == SCHED_OLDEST_SPLIT_WARP) || (i == SCHED_YOUNGEST_SPLIT_WARP))
		{
			if (dRLSched->numSplitWarps == 0)
				continue;
		}
		else if ((i == SCHED_OLDEST_LONG_LAT_MEM_WARP) || (i == SCHED_YOUNGEST_LONG_LAT_MEM_WARP))
		{
			if (dRLSched->readyLongLatMemInstrs == false)
				continue;
		}

		unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
     	unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
     	float lQvalue = mGetQvalue(lIndex);
		//if (smId == 0)
			//printf("%u:%f ", i, lQvalue);
		if (lPrimaryMaxQAction == 0xdeaddead)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
		else if (lQvalue > lPrimaryMaxQvalue)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
	}
	//if (smId == 0)
		//printf("\n");
	action = lPrimaryMaxQAction;
	return action;
}

unsigned int rlEngine::mGetWhichWarpTypeAction(bool exploration, long int randVal)
{
	unsigned int action = 0xdeaddead;
    //unsigned int smId = dRLSched->m_shader->get_sid();
	//unsigned int schedId = dRLSched->m_id;
	if (exploration)
	{
		uint maxNumActions = MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE;

		bool done = false;
		while (done == false)
		{
			action = randVal % maxNumActions;
	
			if (action == SCHED_YOUNGEST_BARRIER_WARP)
			{
				if (gBarrierTbIdSet.empty())
				{
        			randVal = random();
					continue;
				}
			}
			else if (action == SCHED_YOUNGEST_FINISH_WARP)
			{
				if (gFinishTbIdSet.empty())
				{
        			randVal = random();
					continue;
				}
			}
			done = true;
		}
		//if (smId == 0)
			//printf("%llu: sched %u, exploration cycle action chosen %u\n", gpu_sim_cycle, schedId, action);
		return action;
	}

	unsigned int lPrimaryMaxQAction = 0xdeaddead;
	float lPrimaryMaxQvalue = 0.0;
	//if (smId == 0)
		//printf("%llu: sched %u q values: ", gpu_sim_cycle,  schedId);
	//for (unsigned int i = 0; i < MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE; i++)

	for (unsigned int j = 0; j < MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE; j++)
	{
		unsigned int i = j;
		if (gBaseSched == 1) //LRR
			i = (j + 2) % MAX_ACTIONS_OF_TYPE_WHICH_WARP_TYPE;
		if (i == SCHED_YOUNGEST_BARRIER_WARP)
		{
			if (gBarrierTbIdSet.empty())
				continue;
		}
		else if (i == SCHED_YOUNGEST_FINISH_WARP)
		{
			if (gFinishTbIdSet.empty())
				continue;
		}

		unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
     	unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
     	float lQvalue = mGetQvalue(lIndex);
		//if (smId == 0)
			//printf("%u:%f ", i, lQvalue);
		if (lPrimaryMaxQAction == 0xdeaddead)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
		else if (lQvalue > lPrimaryMaxQvalue)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
	}
	action = lPrimaryMaxQAction;
	//if (smId == 0)
		//printf("action chosen %u\n", action);
	return action;
}

unsigned int rlEngine::mGetLrrGtoAction(bool exploration, long int randVal)
{
	unsigned int action = 0xdeaddead;
    //unsigned int smId = dRLSched->m_shader->get_sid();
	//unsigned int schedId = dRLSched->m_id;
	if (exploration)
	{
		action = randVal % MAX_ACTIONS_OF_LRR_GTO_TYPE;

		//if (smId == 0)
			//printf("%llu: sched %u, exploration cycle action chosen %u\n", gpu_sim_cycle, schedId, action);
		return action;
	}

	unsigned int lPrimaryMaxQAction = 0xdeaddead;
	float lPrimaryMaxQvalue = 0.0;
	//if (smId == 0)
		//printf("%llu: sched %u q values: ", gpu_sim_cycle,  schedId);

	for (unsigned int i = 0; i < MAX_ACTIONS_OF_LRR_GTO_TYPE; i++)
	{
		unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
     	unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
     	float lQvalue = mGetQvalue(lIndex);
		//if (smId == 0)
			//printf("%u:%f ", i, lQvalue);
		if (lPrimaryMaxQAction == 0xdeaddead)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
		else if (lQvalue > lPrimaryMaxQvalue)
		{
			lPrimaryMaxQAction = i;
			lPrimaryMaxQvalue = lQvalue;
		}
	}
	action = lPrimaryMaxQAction;
	//if (smId == 0)
		//printf("action chosen %u\n", action);
	return action;
}

unsigned int rlEngine::mGetNumWarpsAction(bool exploration, long int randVal, unsigned int& numOfSWLwarps)
{
	unsigned int action = 0xdeaddead;
	unsigned int numWarps = 32;
	unsigned int maxNumPossibleActions = MAX_ACTIONS_OF_TYPE_NUM_WARPS;

	if (gMaxNumResidentWarpsPerSched <= 16)
		maxNumPossibleActions--;
	if (gMaxNumResidentWarpsPerSched <= 8)
		maxNumPossibleActions--;
	if (gMaxNumResidentWarpsPerSched <= 4)
		maxNumPossibleActions--;
	if (gMaxNumResidentWarpsPerSched <= 2)
		maxNumPossibleActions--;
	if (gMaxNumResidentWarpsPerSched <= 1)
		maxNumPossibleActions--;

	if (exploration || dFirstTime)
	{
		action = randVal % maxNumPossibleActions;
	}
	else
	{
		unsigned int lPrimaryMaxQAction = 0xdeaddead;
		float lPrimaryMaxQvalue = 0.0;
		for (unsigned int i = 0; i < maxNumPossibleActions; i++)
		{
			unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
      		unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
      		float lQvalue = mGetQvalue(lIndex);
			if (lPrimaryMaxQAction == 0xdeaddead)
			{
				lPrimaryMaxQAction = i;
				lPrimaryMaxQvalue = lQvalue;
			}
			else if (lQvalue > lPrimaryMaxQvalue)
			{
				lPrimaryMaxQAction = i;
				lPrimaryMaxQvalue = lQvalue;
			}
		}
		action = lPrimaryMaxQAction;
	}

	// action = SCHED_ALL_WARPS; //JAYVANT to disable this functionality

	assert(action != 0xdeaddead);

	if (action == SCHED_ONE_WARP)
		numWarps = 1;
	else if (action == SCHED_TWO_WARPS)
		numWarps = 2;
	else if (action == SCHED_FOUR_WARPS)
		numWarps = 4;
	else if (action == SCHED_EIGHT_WARPS)
		numWarps = 8;
	else if (action == SCHED_SIXTEEN_WARPS)
		numWarps = 16;
	else if (action == SCHED_ALL_WARPS)
		numWarps = 32;
	if (numWarps > gMaxNumResidentWarpsPerSched)
		numWarps = gMaxNumResidentWarpsPerSched;

	if (numWarps != numOfSWLwarps)
	{
		// printf("%llu: sm %u, sched %u, limiting num of warps to %u\n", gpu_sim_cycle, smId, this->m_id, numWarps);
		numOfSWLwarps = numWarps;
	}

	return action;
}

unsigned int rl_scheduler::mGetNumWarpsAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetNumWarpsAction(exploration, randVal, dNumOfSWLwarps);
}

unsigned int rlEngine::mGetBypassAction(bool exploration, long int randVal)
{
	unsigned int action = 0xdeaddead;

	if (exploration || dFirstTime)
	{
		action = randVal % MAX_ACTIONS_OF_TYPE_BYPASS_L1;
	}
	else
	{
		unsigned int lPrimaryMaxQAction = 0xdeaddead;
		float lPrimaryMaxQvalue = 0.0;
		for (unsigned int i = 0; i < MAX_ACTIONS_OF_TYPE_BYPASS_L1; i++)
		{
			unsigned long long lPrimaryStateVal = getCurrStateForAction(i);
       		unsigned long long lIndex = (lPrimaryStateVal * dNumActions) + i;
       		float lQvalue = mGetQvalue(lIndex);
			if (lPrimaryMaxQAction == 0xdeaddead)
			{
				lPrimaryMaxQAction = i;
				lPrimaryMaxQvalue = lQvalue;
			}
			else if (lQvalue > lPrimaryMaxQvalue)
			{
				lPrimaryMaxQAction = i;
				lPrimaryMaxQvalue = lQvalue;
			}
		}
		action = lPrimaryMaxQAction;
	}

	assert(action != 0xdeaddead);
	
	// action = DO_NOT_BYPASS_L1;//JAYVANT to disable this functionality
	return action;
}

unsigned int rl_scheduler::mGetWhichSchedAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetWhichSchedAction(exploration, randVal);
}

unsigned int rl_scheduler::mGetWhichWarpAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetWhichWarpAction(exploration, randVal);
}

unsigned int rl_scheduler::mGetWhichWarpTypeAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetWhichWarpTypeAction(exploration, randVal);
}

unsigned int rl_scheduler::mGetLrrGtoAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetLrrGtoAction(exploration, randVal);
}

unsigned int rl_scheduler::mGetBypassAction(bool exploration, long int randVal)
{
	return dRLEngines[0]->mGetBypassAction(exploration, randVal);
}

unsigned int rl_scheduler::mGetMaxQaction(std::set<unsigned int>& xPossibleActionSet, unsigned int xMaxNumActions, unsigned int xEngineNum)
{
	unsigned int lMaxQaction = 0xdeaddead;
	float lMaxQvalue = 0.0;
	//unsigned int lNumPossibleActions = xPossibleActionSet.size();
	for (unsigned int i = 0; i < xMaxNumActions; i++)
	{
		if (xPossibleActionSet.find(i) != xPossibleActionSet.end())
		{
			unsigned long long lStateVal = dRLEngines[xEngineNum]->getCurrStateForAction(i);
        	unsigned long long lIndex = (lStateVal * xMaxNumActions) + i;
        	float lQvalue =  dRLEngines[xEngineNum]->mGetQvalue(lIndex);
			if (lMaxQaction == 0xdeaddead)
			{
				lMaxQaction = i;
				lMaxQvalue = lQvalue;
			}
			else if (gUseMinAction == false)
			{
				if (lQvalue > lMaxQvalue)
				{
					lMaxQaction = i;
					lMaxQvalue = lQvalue;
				}
			}
			else 
			{
				if (lQvalue < lMaxQvalue)
				{
					lMaxQaction = i;
					lMaxQvalue = lQvalue;
				}
			}
		}
	}
	return lMaxQaction;
}

unsigned int rl_scheduler::mGetMaxQactionNewActorCritic(std::set<unsigned int>& xPossibleActionSet, unsigned int xMaxNumActions, unsigned int xEngineNum)
{
	return mGetSoftmaxAction(xPossibleActionSet, xMaxNumActions, xEngineNum, false/*Hvalue*/);
}

unsigned int rl_scheduler::mGetMaxHaction(std::set<unsigned int>& xPossibleActionSet, unsigned int xMaxNumActions, unsigned int xEngineNum)
{
	return mGetSoftmaxAction(xPossibleActionSet, xMaxNumActions, xEngineNum, true/*Hvalue*/);
}

unsigned int rl_scheduler::mGetSoftmaxAction(std::set<unsigned int>& xPossibleActionSet, unsigned int xMaxNumActions, unsigned int xEngineNum, bool Hvalue)
{
	float sumWts = 0.0;
	//unsigned int lNumPossibleActions = xPossibleActionSet.size();
	for (unsigned int i = 0; i < xMaxNumActions; i++)
	{
		if (xPossibleActionSet.find(i) != xPossibleActionSet.end())
		{
			unsigned long long lStateVal = dRLEngines[xEngineNum]->getCurrStateForAction(i);
        	unsigned long long lIndex = (lStateVal * xMaxNumActions) + i;
			float hqVal;
			if (Hvalue)
				hqVal = dRLEngines[xEngineNum]->mGetHvalue(lIndex);
			else
				hqVal = dRLEngines[xEngineNum]->mGetQvalue(lIndex);
			float wt = exp(hqVal);
			sumWts += wt;
		}
	}
    long int randVal = random();
    float prob = ((float)randVal/(float)RAND_MAX) * sumWts;
	float cumWt = 0.0;

	unsigned int lMaxAction = 0xdeaddead;
	for (unsigned int i = 0; i < xMaxNumActions; i++)
	{
		if (xPossibleActionSet.find(i) != xPossibleActionSet.end())
		{
			unsigned long long lStateVal = dRLEngines[xEngineNum]->getCurrStateForAction(i);
        	unsigned long long lIndex = (lStateVal * xMaxNumActions) + i;
			float hqVal;
			if (Hvalue)
				hqVal = dRLEngines[xEngineNum]->mGetHvalue(lIndex);
			else
				hqVal = dRLEngines[xEngineNum]->mGetQvalue(lIndex);
			float wt = exp(hqVal);
			cumWt += wt;
			if (prob < cumWt)
				lMaxAction = i;
		}
	}
	return lMaxAction;
}

void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / (1024.0 * 1024.0);
   resident_set = (rss * page_size_kb)/1024.0;
}

void rl_scheduler::order_warps()
{
	if ((gpu_sim_cycle - gBetaUpdatedAtCycle) == 1000)
	{
		gBetaUpdatedAtCycle = gpu_sim_cycle;
		gBeta = 0.99 * gBeta;
	}
	if (gpu_sim_cycle - gLastPrintCycle == 10000)
	{
		gLastPrintCycle = gpu_sim_cycle;
		//printf("gTotalGPUWideReward = %d\n", gTotalGPUWideReward);
		//printf("num rewards per 500 cycles = %d\n", gTotalGPUWideReward - gTotalGPUWideRewardPrev);
		gTotalGPUWideRewardPrev = gTotalGPUWideReward;
		double vmUsage, res;
		process_mem_usage(vmUsage, res);
		printf("process mem usage: vm %f MB, res %f MB\n", vmUsage, res);
	}

	if (dRLEngines[0]->dAttributeVector.size() == 0)
		return;

    m_next_cycle_prioritized_warps.clear();

    unsigned int smId = m_shader->get_sid();
    assert(gNumSMs > smId);

    gReadyTBIdSet.clear();

    initRLAttributeArrays(smId);

    setAttributeValues2();

    computeCurrState();

    std::vector<shd_warp_t*> warpVec;

	//warpVec = m_supervised_warps;

	if (gBaseSched == 0) //GTO
	{
    	order_by_priority( warpVec,
                       	m_supervised_warps,
                       	m_last_supervised_issued,
                       	m_supervised_warps.size(),
                       	ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                       	scheduler_unit::sort_warps_by_oldest_dynamic_id );
	}
	else if (gBaseSched == 1) //LRR
	{
    	order_lrr( warpVec,
               	m_supervised_warps,
               	m_last_supervised_issued,
               	m_supervised_warps.size() );
	}
	else if (gBaseSched == 2) //OLDEST
	{
    	order_by_priority( warpVec,
                       	m_supervised_warps,
                       	m_last_supervised_issued,
                       	m_supervised_warps.size(),
                       	ORDERED_PRIORITY_FUNC_ONLY,
                       	scheduler_unit::sort_warps_by_oldest_dynamic_id );
	}
	else if (gBaseSched == 3) //RANDOM
	{
        long int randVal = random();
        unsigned int idx = randVal % m_supervised_warps.size();
        for (unsigned i = idx; i < m_supervised_warps.size(); ++i)
            warpVec.push_back(m_supervised_warps[i]);
        for (unsigned i = 0; i < idx; ++i)
            warpVec.push_back(m_supervised_warps[i]);
	}
	else
		assert(0);


	gGTOWarpOrder = warpVec;

	gBarrierTbIdSet.clear();
	gFinishTbIdSet.clear();

    if (/*(gReadyTBIdSet.size() > 0) &&*/ (warpVec.size() > 0))
    {
        long int randVal = random();
        float rVal = (randVal / (float) RAND_MAX) * 100;
        bool exploration = (rVal < dCurrExplorationPercent) ? true : false;

		if (gUseWhichSchedAsAction)
		{
            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;

			unsigned int action = dRLEngines[0]->dCurrAction;

        	std::map<unsigned int, unsigned int>& lNumWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtBarrierMap.begin();
				 	iter != lNumWarpsAtBarrierMap.end();
			  	iter++)
			{
				unsigned int lNumWarpsAtBarrier = iter->second;
				if (lNumWarpsAtBarrier > 0)
					gBarrierTbIdSet.insert(iter->first);
			}
				
        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
				 iter != lNumWarpsAtFinishMap.end();
			  	 iter++)
			{
				unsigned int lNumWarpsAtFinish = iter->second;
				if (lNumWarpsAtFinish > 0)
					gFinishTbIdSet.insert(iter->first);
			}

			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				action = mGetWhichSchedAction(exploration, randVal);

				if (gGTOWarpAsAction)
					action = SCHED_GTO_WARP;
				else if (gLRRWarpAsAction)
					action = SCHED_LRR_WARP;
				else if (gYFBWarpAsAction)
					action = SCHED_YFB_WARP;
				else if (gMFSWarpAsAction)
					action = SCHED_MFS_WARP;
				else if (gFMSWarpAsAction)
					action = SCHED_FMS_WARP;

				dRLEngines[0]->dCurrAction = action;
				//if (smId == 0)
					//printf("%llu:action = %u\n", gpu_sim_cycle, action);
			}
			assert(action < dRLEngines[0]->dNumActions);
			gPrimaryActionCntMap[action]++;
			takePrimaryActionCntSnapshot();

			if (action == SCHED_GTO_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                        			scheduler_unit::sort_warps_by_oldest_dynamic_id);
			}
			else if (action == SCHED_LRR_WARP)
			{
    			order_lrr(m_next_cycle_prioritized_warps,
               		  	m_supervised_warps,
               		  	m_last_supervised_issued,
               		  	m_supervised_warps.size());
			}
			else if (action == SCHED_YFB_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                       				scheduler_unit::sort_warps_by_yfb_dynamic_id);
			}
			else if (action == SCHED_MFS_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                       				scheduler_unit::sort_warps_by_mfs_cmd_type);
			}
			else if (action == SCHED_FMS_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                       				scheduler_unit::sort_warps_by_fms_cmd_type);
			}
			else
				assert(0);

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;
		}
		else if (gUseWhichWarpAsAction)
		{
            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;

			unsigned int action = dRLEngines[0]->dCurrAction;

        	std::map<unsigned int, unsigned int>& lNumWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtBarrierMap.begin();
				 iter != lNumWarpsAtBarrierMap.end();
			  	 iter++)
			{
				unsigned int lNumWarpsAtBarrier = iter->second;
				if (lNumWarpsAtBarrier > 0)
					gBarrierTbIdSet.insert(iter->first);
			}
				
        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
				 iter != lNumWarpsAtFinishMap.end();
			  	 iter++)
			{
				unsigned int lNumWarpsAtFinish = iter->second;
				if (lNumWarpsAtFinish > 0)
					gFinishTbIdSet.insert(iter->first);
			}

			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				if (gReadyTBIdSet.size() != 0)
				{
					action = mGetWhichWarpAction(exploration, randVal);

					dRLEngines[0]->dCurrAction = action;
				}
				//if (smId == 0)
					//printf("%llu:action = %u\n", gpu_sim_cycle, action);
			}
			assert(action < dRLEngines[0]->dNumActions);
			gPrimaryActionCntMap[action]++;
			takePrimaryActionCntSnapshot();

			if (action == SCHED_GTO_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				//ORDERED_PRIORITY_FUNC_ONLY,
                        			//scheduler_unit::sort_warps_by_oldest_dynamic_id);
                       				ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                        			scheduler_unit::sort_warps_by_oldest_dynamic_id);
			}
			else if (action == SCHED_YOUNGEST_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_youngest_dynamic_id);
			}
			else if (action == SCHED_NEXT_WARP)
			{
    			order_lrr(m_next_cycle_prioritized_warps,
               		  	m_supervised_warps,
               		  	m_last_supervised_issued,
               		  	m_supervised_warps.size());
			}
			else if (action == SCHED_YOUNGEST_BARRIER_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_barrier_flag);
			}
			else if (action == SCHED_YOUNGEST_FINISH_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_finish_flag);
			}
			else if (action == SCHED_OLDEST_SPLIT_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_split_oldest_dynamic_id);
			}
			else if (action == SCHED_YOUNGEST_SPLIT_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_split_youngest_dynamic_id);
			}
			else if (action == SCHED_OLDEST_LONG_LAT_MEM_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_long_lat_mem_oldest_dynamic_id);
			}
			else if (action == SCHED_YOUNGEST_LONG_LAT_MEM_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_long_lat_mem_youngest_dynamic_id);
			}
			else
				assert(0);

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;
		}
		else if (gUseWhichWarpTypeAsAction)
		{
            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;

			unsigned int action = dRLEngines[0]->dCurrAction;

        	std::map<unsigned int, unsigned int>& lNumWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtBarrierMap.begin();
				 iter != lNumWarpsAtBarrierMap.end();
			  	 iter++)
			{
				unsigned int lNumWarpsAtBarrier = iter->second;
				if (lNumWarpsAtBarrier > 0)
					gBarrierTbIdSet.insert(iter->first);
			}
				
        	std::map<unsigned int, unsigned int>& lNumWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(smId);
			for (std::map<unsigned int, unsigned int>::iterator iter = lNumWarpsAtFinishMap.begin();
				 iter != lNumWarpsAtFinishMap.end();
			  	 iter++)
			{
				unsigned int lNumWarpsAtFinish = iter->second;
				if (lNumWarpsAtFinish > 0)
					gFinishTbIdSet.insert(iter->first);
			}

			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				//if ((smId == 0) && exploration)
					//printf("%llu: Exploration Cycle\n", gpu_sim_cycle);
				if (gReadyTBIdSet.size() != 0)
				{
					action = mGetWhichWarpTypeAction(exploration, randVal);
					dRLEngines[0]->dCurrAction = action;
				}
				//else if (smId == 0)
				//{
					//unsigned int schedId = this->m_id;
					//printf("%llu: sched %u No schedulable warps\n", gpu_sim_cycle, schedId);
				//}
			}
			assert(action < dRLEngines[0]->dNumActions);
			gPrimaryActionCntMap[action]++;
			takePrimaryActionCntSnapshot();

			if (action == SCHED_GTO_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				//ORDERED_PRIORITY_FUNC_ONLY,
                        			//scheduler_unit::sort_warps_by_oldest_dynamic_id);
                       				ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                        			scheduler_unit::sort_warps_by_oldest_dynamic_id);
			}
			else if (action == SCHED_YOUNGEST_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_youngest_dynamic_id);
			}
			else if (action == SCHED_NEXT_WARP)
			{
    			order_lrr(m_next_cycle_prioritized_warps,
               		  	m_supervised_warps,
               		  	m_last_supervised_issued,
               		  	m_supervised_warps.size());
			}
			else if (action == SCHED_YOUNGEST_BARRIER_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_barrier_flag);
			}
			else if (action == SCHED_YOUNGEST_FINISH_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				ORDERED_PRIORITY_FUNC_ONLY,
                        			scheduler_unit::sort_warps_by_finish_flag);
			}
			else
				assert(0);

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;
		}
		else if (gUseLrrGtoAsAction)
		{
            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;

			unsigned int action = dRLEngines[0]->dCurrAction;

			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				//if ((smId == 0) && exploration)
					//printf("%llu: Exploration Cycle\n", gpu_sim_cycle);
				action = mGetLrrGtoAction(exploration, randVal);
				dRLEngines[0]->dCurrAction = action;
				//else if (smId == 0)
				//{
					//unsigned int schedId = this->m_id;
					//printf("%llu: sched %u No schedulable warps\n", gpu_sim_cycle, schedId);
				//}
			}
			assert(action < dRLEngines[0]->dNumActions);
			gPrimaryActionCntMap[action]++;
			takePrimaryActionCntSnapshot();

			if (action == SCHED_GTO_WARP)
			{
    			order_by_priority(m_next_cycle_prioritized_warps,
                      				m_supervised_warps,
                       				m_last_supervised_issued,
                       				m_supervised_warps.size(),
                       				//ORDERED_PRIORITY_FUNC_ONLY,
                        			//scheduler_unit::sort_warps_by_oldest_dynamic_id);
                       				ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                        			scheduler_unit::sort_warps_by_oldest_dynamic_id);
			}
			else if (action == SCHED_LRR_WARP)
			{
    			order_lrr(m_next_cycle_prioritized_warps,
               		  	m_supervised_warps,
               		  	m_last_supervised_issued,
               		  	m_supervised_warps.size());
			}
			else
				assert(0);

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;
		}
		else if (gUseNumOfWarpsAsAction && gUseBypassL1AsAction)
		{
			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				unsigned int action1 = mGetNumWarpsAction(exploration, randVal);
				unsigned int action2 = mGetBypassAction(exploration, randVal);

				dRLEngines[0]->dCurrAction = (action1 << 1) | action2;
			}

			uint numWarpsAdded = 0;
    		for (unsigned i = 0; i < warpVec.size(); ++i)
			{
        		m_next_cycle_prioritized_warps.push_back(warpVec[i]);
				if (warpVec[i] && (warpVec[i]->done_exit() == false) && (warpVec[i]->waiting() == false))
				{
					numWarpsAdded++;
					if (numWarpsAdded >= dNumOfSWLwarps)
						break;
				}
			}

			if ((dRLEngines[0]->dCurrAction & 0x1) == DO_NOT_BYPASS_L1)
				gBypassL1Cache = false;
			else
				gBypassL1Cache = true;
		}
		else if (gUseNumOfWarpsAsAction)
		{
			unsigned int lSchedFreq = 1;
			if (gSchedFreq == 1)
				lSchedFreq = 32;
			else if (gSchedFreq == 2)
				lSchedFreq = 64;
			else if (gSchedFreq == 4)
				lSchedFreq = 128;
			else if (gSchedFreq == 8)
				lSchedFreq = 256;
			else if (gSchedFreq == 16)
				lSchedFreq = 512;
			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % lSchedFreq) == 0))
			{
				unsigned int action = mGetNumWarpsAction(exploration, randVal);
				dRLEngines[0]->dCurrAction = action;
			}

			uint numWarpsAdded = 0;
    		for (unsigned i = 0; i < warpVec.size(); ++i)
			{
        		m_next_cycle_prioritized_warps.push_back(warpVec[i]);
				if (warpVec[i] && (warpVec[i]->done_exit() == false) && (warpVec[i]->waiting() == false))
				{
					numWarpsAdded++;
					if (numWarpsAdded >= dNumOfSWLwarps)
						break;
				}
			}
		}
		else if (gUseBypassL1AsAction)
		{
			if (dRLEngines[0]->dFirstTime || ((gpu_sim_cycle % gSchedFreq) == 0))
			{
				unsigned int action = mGetBypassAction(exploration, randVal);
				dRLEngines[0]->dCurrAction = action;
			}

    		for (unsigned i = 0; i < warpVec.size(); ++i)
        		m_next_cycle_prioritized_warps.push_back(warpVec[i]);

			if ((dRLEngines[0]->dCurrAction & 0x1) == DO_NOT_BYPASS_L1)
				gBypassL1Cache = false;
			else
				gBypassL1Cache = true;
		}
		else if (gUseNAMaction)
		{
            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;
			std::set<unsigned int> lPossibleActionSet;
            for (std::vector< shd_warp_t* >::iterator iter = warpVec.begin(); iter != warpVec.end(); iter++)
			{
				shd_warp_t* warpPtr = (*iter);
				unsigned int lWarpAction = mGetActionValue(warpPtr, USE_NAM_ACTION);
				if (lWarpAction < MAX_ACTIONS_OF_TYPE_NAM)
					lPossibleActionSet.insert(lWarpAction);
			}

			unsigned int lPrimaryMaxQAction = 0xdeaddead;
			if (exploration || dRLEngines[0]->dFirstTime)
			{
				unsigned int numPossibleActions = lPossibleActionSet.size();
				if (numPossibleActions != 0)
				{
					unsigned int actionIndex = randVal % numPossibleActions;
					unsigned int curr = 0;
					for (std::set<unsigned int>::iterator iter = lPossibleActionSet.begin(); iter != lPossibleActionSet.end(); iter++)
					{
						if (curr == actionIndex)
						{
							lPrimaryMaxQAction = (*iter);
							break;
						}
					}
				}
				else
					lPrimaryMaxQAction = SCHED_NO_INSTR;
            	rl_scheduler::gExplorationCnt++;
			}
			else
			{
				float lPrimaryMaxQvalue = 0.0;
				for (unsigned int i = 0; i < MAX_ACTIONS_OF_TYPE_NAM; i++)
				{
					if (lPossibleActionSet.find(i) != lPossibleActionSet.end())
					{
						unsigned long long lPrimaryStateVal = dRLEngines[0]->getCurrStateForAction(i);
        				unsigned long long lIndex = (lPrimaryStateVal * dRLEngines[0]->dNumActions) + i;
        				float lQvalue =  dRLEngines[0]->mGetQvalue(lIndex);
						if (lPrimaryMaxQAction == 0xdeaddead)
						{
							lPrimaryMaxQAction = i;
							lPrimaryMaxQvalue = lQvalue;
						}
						else if (gUseMinAction == false)
						{
							if (lQvalue > lPrimaryMaxQvalue)
							{
								lPrimaryMaxQAction = i;
								lPrimaryMaxQvalue = lQvalue;
							}
						}
						else 
						{
							if (lQvalue < lPrimaryMaxQvalue)
							{
								lPrimaryMaxQAction = i;
								lPrimaryMaxQvalue = lQvalue;
							}
						}
					}
				}
			}
			if (lPrimaryMaxQAction == 0xdeaddead)
				lPrimaryMaxQAction = SCHED_NO_INSTR;
			dRLEngines[0]->dCurrAction = lPrimaryMaxQAction;
			std::vector<shd_warp_t*> vec1;
            for (std::vector< shd_warp_t* >::iterator iter = warpVec.begin(); iter != warpVec.end(); iter++)
			{
				shd_warp_t* warpPtr = (*iter);
				unsigned int lWarpAction = mGetActionValue(warpPtr, USE_NAM_ACTION);
				if (lWarpAction == lPrimaryMaxQAction)
                 	m_next_cycle_prioritized_warps.push_back(warpPtr);
				else
					vec1.push_back(warpPtr);
			}
            for (std::vector< shd_warp_t* >::iterator iter = vec1.begin(); iter != vec1.end(); iter++)
			{
				shd_warp_t* warpPtr = (*iter);
                m_next_cycle_prioritized_warps.push_back(warpPtr);
			}

			gPrimaryActionCntMap[lPrimaryMaxQAction]++;
			takePrimaryActionCntSnapshot();

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;
		}
        else if (dRLEngines[0]->dFirstTime || 
		    (exploration && (rl_scheduler::gUseCMACFuncApprox == false)))
        {
            rl_scheduler::gExplorationCnt++;

			if (gUseCmdPipeTbTypeNumWarpsBypassAsAction)
			{
				unsigned int action = dRLEngines[2]->mGetNumWarpsAction(exploration, randVal, dNumOfSWLwarps);
				dRLEngines[2]->dCurrAction = action;

    			std::vector<shd_warp_t*> warpVec1 = warpVec;
				warpVec.clear();

				uint numWarpsAdded = 0;
    			for (unsigned i = 0; i < m_supervised_warps.size(); ++i)
				{
        			warpVec.push_back(m_supervised_warps[i]);
					if (m_supervised_warps[i] && (m_supervised_warps[i]->done_exit() == false) && (m_supervised_warps[i]->waiting() == false))
					{
						numWarpsAdded++;

						if (numWarpsAdded >= dNumOfSWLwarps)
							break;
					}
				}

            	unsigned int idx = randVal % warpVec.size();
            	for (unsigned i = idx; i < warpVec.size(); ++i)
                	m_next_cycle_prioritized_warps.push_back(warpVec[i]);
            	for (unsigned i = 0; i < idx; ++i)
                	m_next_cycle_prioritized_warps.push_back(warpVec[i]);

				action = dRLEngines[3]->mGetBypassAction(exploration, randVal);
				dRLEngines[3]->dCurrAction = action;

				if ((dRLEngines[3]->dCurrAction & 0x1) == DO_NOT_BYPASS_L1)
					gBypassL1Cache = false;
				else
					gBypassL1Cache = true;
			}
			/*
			else if ((rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION) || 
					 (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION))
			{
				unsigned int numActions = 0;
				if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
					numActions = MAX_ACTIONS_OF_TYPE_CMD_PIPE;
				else //if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
					numActions = MAX_ACTIONS_OF_TYPE_TB_TYPE;
				unsigned int idx = randVal % lNumPrimaryActions;
				for (int i = 0; i < numActions; i++)
				{
				}
			}
			*/
			else
			{
            	unsigned int idx = randVal % m_supervised_warps.size();
            	for (unsigned i = idx; i < m_supervised_warps.size(); ++i)
                	m_next_cycle_prioritized_warps.push_back(m_supervised_warps[i]);
            	for (unsigned i = 0; i < idx; ++i)
                	m_next_cycle_prioritized_warps.push_back(m_supervised_warps[i]);
			}
        }
        else
        {
			if (gUseCmdPipeTbTypeNumWarpsBypassAsAction)
			{
				unsigned int action = dRLEngines[2]->mGetNumWarpsAction(exploration, randVal, dNumOfSWLwarps);
				dRLEngines[2]->dCurrAction = action;

    			std::vector<shd_warp_t*> warpVec1 = warpVec;
				warpVec.clear();

				uint numWarpsAdded = 0;
    			for (unsigned i = 0; i < warpVec1.size(); ++i)
				{
        			warpVec.push_back(warpVec1[i]);
					if (warpVec1[i] && (warpVec1[i]->done_exit() == false) && (warpVec1[i]->waiting() == false))
					{
						numWarpsAdded++;

						if (numWarpsAdded >= dNumOfSWLwarps)
							break;
					}
				}

				action = dRLEngines[3]->mGetBypassAction(exploration, randVal);
				dRLEngines[3]->dCurrAction = action;

				if ((dRLEngines[3]->dCurrAction & 0x1) == DO_NOT_BYPASS_L1)
					gBypassL1Cache = false;
				else
					gBypassL1Cache = true;
			}

			gExploitationCnt++;
            gPrimaryRLEngine = dRLEngines[0];
            rl_scheduler::gPrimaryQvalues = dRLEngines[0]->dValues;
            rl_scheduler::gPrimaryStateVal = dRLEngines[0]->dCurrState;
            rl_scheduler::gPrimaryNumActions = dRLEngines[0]->dNumActions;

			if (dRLEngines[1])
			{
            	gSecondaryRLEngine = dRLEngines[1];
            	rl_scheduler::gSecondaryQvalues = dRLEngines[1]->dValues;
            	rl_scheduler::gSecondaryStateVal = dRLEngines[1]->dCurrState;
            	rl_scheduler::gSecondaryNumActions = dRLEngines[1]->dNumActions;
			}

            rl_scheduler::gScoreboard = m_scoreboard;
            rl_scheduler::gSimtStack = m_simt_stack;
			rl_scheduler::gCurrRLSchedulerUnit = this;
    
			if (rl_scheduler::gUseCMACFuncApprox)
			{
				double lBestValueRet = 0.0;
				unsigned int lPrimaryMaxQAction =  dRLEngines[0]->dSarsaAgent->selectAction(dRLEngines[0]->dCurrStateVector, lBestValueRet);
				gPrimaryActionCntMap[lPrimaryMaxQAction]++;
				takePrimaryActionCntSnapshot();
				gPrintQvalues = false;

				std::vector<shd_warp_t*> vec1;
				std::vector<shd_warp_t*> vec2;
				std::vector<shd_warp_t*> vec3;

				unsigned int lSecondaryMaxQAction = 0xdeaddead;
				if (dRLEngines[1])
					lSecondaryMaxQAction = dRLEngines[1]->dSarsaAgent->selectAction(dRLEngines[1]->dCurrStateVector);

               	std::vector< shd_warp_t* >::iterator iter;

				//add warps matching primary max Q action and secondary max Q action to m_next_cycle_prioritized_warps
				//matching only primary max Q action to vec1 and remaining to vec2

               	for (iter = warpVec.begin(); iter != warpVec.end(); iter++)
				{
					shd_warp_t* warpPtr = (*iter);
					unsigned int lWarpAction = mGetActionValue(warpPtr, rl_scheduler::dRLActionTypes[0]);
					if (lWarpAction == lPrimaryMaxQAction)
					{
						if (dRLEngines[1])
						{
							lWarpAction = mGetActionValue(warpPtr, rl_scheduler::dRLActionTypes[1]);
							if (lWarpAction == lSecondaryMaxQAction)
                   				m_next_cycle_prioritized_warps.push_back(warpPtr);
							else
                   				vec1.push_back(warpPtr);
						}
						else
						{
                   			m_next_cycle_prioritized_warps.push_back(warpPtr);
						}
					}
					else if (lWarpAction == lSecondaryMaxQAction)
						vec2.push_back(warpPtr);
					else
						vec3.push_back(warpPtr);
				}

               	for (iter = vec1.begin(); iter != vec1.end(); iter++)
				{
					shd_warp_t* warpPtr = (*iter);
                   	m_next_cycle_prioritized_warps.push_back(warpPtr);
				}

               	for (iter = vec2.begin(); iter != vec2.end(); iter++)
				{
					shd_warp_t* warpPtr = (*iter);
                   	m_next_cycle_prioritized_warps.push_back(warpPtr);
				}

               	for (iter = vec3.begin(); iter != vec3.end(); iter++)
				{
					shd_warp_t* warpPtr = (*iter);
                   	m_next_cycle_prioritized_warps.push_back(warpPtr);
				}
			}
			else if ((rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION) || 
					 (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION))
			{
				std::set<unsigned int> lPossiblePrimaryActionSet;
				std::set<unsigned int> lPossibleSecondaryActionSet;

				unsigned int lNumPrimaryActions = 0xdeaddead;
				unsigned int lNumSecondaryActions = 0xdeaddead;
				if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
				{
					lNumPrimaryActions = MAX_ACTIONS_OF_TYPE_CMD_PIPE;
					if (dConsecutiveNoInstrSchedCnt < gMaxConsNoInstrs)
						lPossiblePrimaryActionSet.insert(SCHED_NO_INSTR);
				}
				else if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
				{
					lNumPrimaryActions = MAX_ACTIONS_OF_TYPE_TB_TYPE;
				}
				else
					assert(0);

				if (dRLEngines[1])
				{
					if (rl_scheduler::dRLActionTypes[1] == USE_CMD_PIPE_AS_ACTION)
					{
						lNumSecondaryActions = MAX_ACTIONS_OF_TYPE_CMD_PIPE;
						if (dConsecutiveNoInstrSchedCnt < gMaxConsNoInstrs)
							lPossibleSecondaryActionSet.insert(SCHED_NO_INSTR);
					}
					else if (rl_scheduler::dRLActionTypes[1] == USE_TB_TYPE_AS_ACTION)
						lNumSecondaryActions = MAX_ACTIONS_OF_TYPE_TB_TYPE;
					else
						assert(0);
				}

               	for (std::vector< shd_warp_t* >::iterator iter = warpVec.begin(); iter != warpVec.end(); iter++)
				{
					shd_warp_t* warpPtr = (*iter);
					unsigned int lWarpAction = mGetActionValue(warpPtr, rl_scheduler::dRLActionTypes[0]);
					if (lWarpAction < lNumPrimaryActions)
						lPossiblePrimaryActionSet.insert(lWarpAction);

					if (dRLEngines[1])
					{
						lWarpAction = mGetActionValue(warpPtr, rl_scheduler::dRLActionTypes[1]);
						if (lWarpAction < lNumSecondaryActions)
							lPossibleSecondaryActionSet.insert(lWarpAction);
					}
				}
				unsigned int lNumPossiblePrimaryActions = lPossiblePrimaryActionSet.size();
				gPossiblePrimaryActionCntMap[lNumPossiblePrimaryActions]++;

				unsigned int lPrimaryAction;
				if (gActorCriticMethod)
					lPrimaryAction = mGetMaxHaction(lPossiblePrimaryActionSet, rl_scheduler::gPrimaryNumActions, 0);
				else if (gNewActorCriticMethod)
					lPrimaryAction = mGetMaxQactionNewActorCritic(lPossiblePrimaryActionSet, rl_scheduler::gPrimaryNumActions, 0);
				else
					lPrimaryAction = mGetMaxQaction(lPossiblePrimaryActionSet, rl_scheduler::gPrimaryNumActions, 0);

				if (lPrimaryAction == 0xdeaddead)
				{
					if (rl_scheduler::dRLActionTypes[0] == USE_CMD_PIPE_AS_ACTION)
						lPrimaryAction = SCHED_NO_TB;
					else if (rl_scheduler::dRLActionTypes[0] == USE_TB_TYPE_AS_ACTION)
						lPrimaryAction = SCHED_NO_INSTR;
				}
				unsigned int lSecondaryAction = 0xdeaddead;
				if (dRLEngines[1])
				{
					if (rl_scheduler::dRLActionTypes[1] == USE_CMD_PIPE_AS_ACTION)
						lNumSecondaryActions = MAX_ACTIONS_OF_TYPE_CMD_PIPE;
					else if (rl_scheduler::dRLActionTypes[1] == USE_TB_TYPE_AS_ACTION)
						lNumSecondaryActions = MAX_ACTIONS_OF_TYPE_TB_TYPE;
					else
						assert(0);

					if (gActorCriticMethod)
						lSecondaryAction = mGetMaxHaction(lPossibleSecondaryActionSet, rl_scheduler::gSecondaryNumActions, 1);
					else if (gNewActorCriticMethod)
						lSecondaryAction = mGetMaxQactionNewActorCritic(lPossibleSecondaryActionSet, rl_scheduler::gSecondaryNumActions, 1);
					else
						lSecondaryAction = mGetMaxQaction(lPossibleSecondaryActionSet, rl_scheduler::gSecondaryNumActions, 1);
					if (lSecondaryAction == 0xdeaddead)
					{
						if (rl_scheduler::dRLActionTypes[1] == USE_CMD_PIPE_AS_ACTION)
							lSecondaryAction = SCHED_NO_TB;
						else if (rl_scheduler::dRLActionTypes[1] == USE_TB_TYPE_AS_ACTION)
							lSecondaryAction = SCHED_NO_INSTR;
					}
				}

				if (gPrimaryActionCntMap.find(lPrimaryAction) == gPrimaryActionCntMap.end())
					gPrimaryActionCntMap[lPrimaryAction] = 0;
				gPrimaryActionCntMap[lPrimaryAction] += 1;
				takePrimaryActionCntSnapshot();

				if (gSecondaryActionCntMap.find(lSecondaryAction) == gSecondaryActionCntMap.end())
					gSecondaryActionCntMap[lSecondaryAction] = 0;
				gSecondaryActionCntMap[lSecondaryAction] += 1;

				// gPrimaryFirstCnt++;

				if ((dFirstMaxQAction == 0xdeaddead) || (gpu_sim_cycle % gSchedFreq) == 0)
				{
					dFirstTime = false;
					dFirstMaxQAction = lPrimaryAction;
					dFirstActionType = rl_scheduler::dRLActionTypes[0];
					dSecondMaxQAction = lSecondaryAction;
					dSecondActionType = rl_scheduler::dRLActionTypes[1];
				}

    			if (gUseTbTypeAsAction)
				{
					if (dFirstMaxQAction == SCHED_FASTEST_TB)
						order_by_priority(warpVec,
                       		m_supervised_warps,
                       		m_last_supervised_issued,
                       		m_supervised_warps.size(),
                       		ORDERED_PRIORITY_FUNC_ONLY,
                       		scheduler_unit::sort_warps_by_oldest_dynamic_id );
					else
						order_by_priority(warpVec,
                       		m_supervised_warps,
                       		m_last_supervised_issued,
                       		m_supervised_warps.size(),
                       		ORDERED_PRIORITY_FUNC_ONLY,
                       		scheduler_unit::sort_warps_by_youngest_dynamic_id );
				}

				if ((dFirstActionType != USE_CMD_PIPE_AS_ACTION) || (dFirstMaxQAction != SCHED_NO_INSTR) || (dConsecutiveNoInstrSchedCnt >= gMaxConsNoInstrs))
				{
					dConsecutiveNoInstrSchedCnt = 0;

					std::vector<shd_warp_t*> vec1;
					std::vector<shd_warp_t*> vec2;
               		for (std::vector< shd_warp_t* >::iterator iter = warpVec.begin(); iter != warpVec.end(); iter++)
					{
						shd_warp_t* warpPtr = (*iter);
						unsigned int lWarpAction = mGetActionValue(warpPtr, dFirstActionType);
						if (lWarpAction == dFirstMaxQAction)
						{
							if (dRLEngines[1])
							{
								lWarpAction = mGetActionValue(warpPtr, dSecondActionType);
								if (lWarpAction == dSecondMaxQAction)
                   					m_next_cycle_prioritized_warps.push_back(warpPtr);
								else
                   					vec1.push_back(warpPtr);
							}
							else
                   				m_next_cycle_prioritized_warps.push_back(warpPtr);
						}
						else
							vec2.push_back(warpPtr);
					}
               		for (std::vector< shd_warp_t* >::iterator iter = vec1.begin(); iter != vec1.end(); iter++)
					{
						shd_warp_t* warpPtr = (*iter);
                   		m_next_cycle_prioritized_warps.push_back(warpPtr);
					}
               		for (std::vector< shd_warp_t* >::iterator iter = vec2.begin(); iter != vec2.end(); iter++)
					{
						shd_warp_t* warpPtr = (*iter);
                   		m_next_cycle_prioritized_warps.push_back(warpPtr);
					}
				}
				else
				{
					dConsecutiveNoInstrSchedCnt++;
				}
			}
			else
			{
				unsigned int lNumWarps = warpVec.size();
				for (unsigned int i = 0; i < (lNumWarps - 1); i++)
				{
					unsigned int lHigherQValueWarpIdx = i;
					for (unsigned int j = (i+1); j < lNumWarps;  j++)
					{
						bool lRetVal = scheduler_unit::sort_warps_by_highest_q_value(warpVec[lHigherQValueWarpIdx], warpVec[j]);
						if (lRetVal == false)
							lHigherQValueWarpIdx = j;
					}
					//swap i and lHigherQValueWarpIdx
					if (i != lHigherQValueWarpIdx)
					{
						shd_warp_t* lTmp  = warpVec[lHigherQValueWarpIdx];
						warpVec[lHigherQValueWarpIdx] = warpVec[i];
						warpVec[i] = lTmp;
					}
				}
    
               	for (unsigned i = 0; i < warpVec.size(); ++i)
                   	m_next_cycle_prioritized_warps.push_back(warpVec[i]);
			}
    
            rl_scheduler::gPrimaryQvalues = 0;
            rl_scheduler::gPrimaryStateVal = 0;
            rl_scheduler::gPrimaryNumActions = 0;

            rl_scheduler::gSecondaryQvalues = 0;
            rl_scheduler::gSecondaryStateVal = 0;
            rl_scheduler::gSecondaryNumActions = 0;

            rl_scheduler::gScoreboard = 0;
            rl_scheduler::gSimtStack = 0;
			rl_scheduler::gCurrRLSchedulerUnit = 0;

        }
    }
}

void rl_scheduler::computeNextValueAndUpdateOldValue(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
	if (gActorCriticMethod)
		computeNextHvalueAndUpdateOldHvalue(warp, pipeUsed, instrSched);
	else
		computeNextQvalueAndUpdateOldQvalue(warp, pipeUsed, instrSched);
}

void rl_scheduler::computeNextHvalueAndUpdateOldHvalue(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->computeNextHvalueAndUpdateOldHvalue(warp, pipeUsed, instrSched);
}

void rl_scheduler::computeNextQvalueAndUpdateOldQvalue(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
	for (unsigned int i = 0; i < dNumRLEngines; i++)
		dRLEngines[i]->computeNextQvalueAndUpdateOldQvalue(warp, pipeUsed, instrSched);
}

void rlEngine::computeNextHvalueAndUpdateOldHvalue(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
	assert (rl_scheduler::gUseCMACFuncApprox == false);
    if ((dFirstTime == false) /*&& lMoreTBsWaiting*/)
    {
        unsigned int action = computeAction(warp, pipeUsed, instrSched);
		dCurrState = getCurrStateForAction(action);
		float Vselected = mGetSvalue(dCurrState);
		if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
		{
        	updateStateActionWeightValuesActorCritic(Vselected);
		}
		else
			updateActorCriticValues(Vselected);

		unsigned int snapShotFreq = 1000;
		if (gpu_sim_cycle <= 25000)
			snapShotFreq = 1000;
		else if (gpu_sim_cycle <= 75000)
			snapShotFreq = 2000;
		else if (gpu_sim_cycle <= 175000)
			snapShotFreq = 4000;
		else if (gpu_sim_cycle <= 375000)
			snapShotFreq = 8000;
		else if (gpu_sim_cycle <= 775000)
			snapShotFreq = 16000;
		else if (gpu_sim_cycle <= 1575000)
			snapShotFreq = 32000;
		else
			snapShotFreq = 64000;
		if ((gpu_sim_cycle % snapShotFreq) == 0)
		{
    		unsigned int smId = dRLSched->m_shader->get_sid();
			unsigned int schedId = dRLSched->m_id;
			if (gSnapshotSmId == 0xdeaddead)
			{
				gSnapshotSmId = smId;
				gSnapshotSchedId = schedId;
			}
			else if ((gSnapshotSmId == smId) && (gSnapshotSchedId == schedId))
			{
				if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
				{
					//printf("weights snapshot:");
					for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
					{
						std::map<unsigned int, std::vector<float> >& lWeightsVecMap = gWeightsVecMapVec[dEngineNum];
						std::vector<float>& wtVec = lWeightsVecMap[i];
						wtVec.push_back(dStateActionWeightArray[i]);
						//printf(" %e", dStateActionWeightArray[i]);
					}
					//printf("\n");
				}
				else
				{
					takeQvalueSnapshot(dValues->dStateActionValueMap, dEngineNum);
					takeQvalueUpdateSnapshot(dValueUpdates->dMap, dEngineNum);
				}
			}
		}
    }
    else
    {
        dFirstTime = false;
    }
}

void rlEngine::takeQvalueAndUpdateSnapshots()
{
    unsigned int snapShotFreq = 1000;
    if (gpu_sim_cycle <= 25000)
        snapShotFreq = 1000;
    else if (gpu_sim_cycle <= 75000)
        snapShotFreq = 2000;
    else if (gpu_sim_cycle <= 175000)
        snapShotFreq = 4000;
    else if (gpu_sim_cycle <= 375000)
        snapShotFreq = 8000;
    else if (gpu_sim_cycle <= 775000)
        snapShotFreq = 16000;
    else if (gpu_sim_cycle <= 1575000)
        snapShotFreq = 32000;
    else
        snapShotFreq = 64000;
    if ((gpu_sim_cycle % snapShotFreq) == 0)
    {
        unsigned int smId = dRLSched->m_shader->get_sid();
        unsigned int schedId = dRLSched->m_id;
        bool takeSnapshot = false;
        if (gSnapshotSmId == 0xdeaddead)
        {
            gSnapshotSmId = smId;
            gSnapshotSchedId = schedId;
            takeSnapshot = true;
        }
        else if ((gSnapshotSmId == smId) && (gSnapshotSchedId == schedId))
        {
            takeSnapshot = true;
        }
        if (takeSnapshot)
        {
            if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
            {
                //printf("weights snapshot:");
                for (unsigned int i = 0; i < (dAttributeVector.size() * dNumActions); i++)
                {
                    std::map<unsigned int, std::vector<float> >& lWeightsVecMap = gWeightsVecMapVec[dEngineNum];
                    std::vector<float>& wtVec = lWeightsVecMap[i];
                    wtVec.push_back(dStateActionWeightArray[i]);
                    //printf(" %e", dStateActionWeightArray[i]);
                }
                //printf("\n");
            }
            else
            {
                takeQvalueSnapshot(dValues->dStateActionValueMap, dEngineNum);
                takeQvalueUpdateSnapshot(dValueUpdates->dMap, dEngineNum);
            }
        }
    }
}

void rlEngine::computeNextQvalueAndUpdateOldQvalue(shd_warp_t* warp, operation_pipeline_t pipeUsed, warp_inst_t* instrSched)
{
    if ((dFirstTime == false) /*&& lMoreTBsWaiting*/)
    {
        unsigned int action = computeAction(warp, pipeUsed, instrSched);

		if (rl_scheduler::gUseCMACFuncApprox)
		{
			dSarsaAgent->setEpsilon(dRLSched->dCurrExplorationPercent);
			dSarsaAgent->setLearningRate(dRLSched->dCurrAlpha);
			dSarsaAgent->update(dCurrStateVector, action, dReward, gGamma);
		}
		else
		{
			dCurrState = getCurrStateForAction(action);
			unsigned long long index = dCurrState * dNumActions + action;
        	float Qselected = mGetQvalue(index);
			if (rl_scheduler::gUseFeatureWeightFuncApprox == true)
        		updateStateActionWeightValues(Qselected);
			else
        		updateQvalue(Qselected);

			takeQvalueAndUpdateSnapshots();
		}
    }
    else
    {
        dFirstTime = false;
    }
}

void shader_core_ctx::read_operands()
{
}

address_type coalesced_segment(address_type addr, unsigned segment_size_lg2bytes)
{
   return  (addr >> segment_size_lg2bytes);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B (32-bit) word
unsigned shader_core_ctx::translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs )
{
   // During functional execution, each thread sees its own memory space for local memory, but these
   // need to be mapped to a shared address space for timing simulation.  We do that mapping here.

   address_type thread_base = 0;
   unsigned max_concurrent_threads=0;
   if (m_config->gpgpu_local_mem_map) {
      // Dnew = D*N + T%nTpC + nTpC*C
      // N = nTpC*nCpS*nS (max concurent threads)
      // C = nS*K + S (hw cta number per gpu)
      // K = T/nTpC   (hw cta number per core)
      // D = data index
      // T = thread
      // nTpC = number of threads per CTA
      // nCpS = number of CTA per shader
      // 
      // for a given local memory address threads in a CTA map to contiguous addresses,
      // then distribute across memory space by CTAs from successive shader cores first, 
      // then by successive CTA in same shader core
      thread_base = 4*(kernel_padded_threads_per_cta * (m_sid + num_shader * (tid / kernel_padded_threads_per_cta))
                       + tid % kernel_padded_threads_per_cta); 
      max_concurrent_threads = kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
   } else {
      // legacy mapping that maps the same address in the local memory space of all threads 
      // to a single contiguous address region 
      thread_base = 4*(m_config->n_thread_per_shader * m_sid + tid);
      max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
   }
   assert( thread_base < 4/*word size*/*max_concurrent_threads );

   // If requested datasize > 4B, split into multiple 4B accesses
   // otherwise do one sub-4 byte memory access
   unsigned num_accesses = 0;

   if(datasize >= 4) {
      // >4B access, split into 4B chunks
      assert(datasize%4 == 0);   // Must be a multiple of 4B
      num_accesses = datasize/4;
      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD); // max 32B
      assert(localaddr%4 == 0); // Address must be 4B aligned - required if accessing 4B per request, otherwise access will overflow into next thread's space
      for(unsigned i=0; i<num_accesses; i++) {
          address_type local_word = localaddr/4 + i;
          address_type linear_address = local_word*max_concurrent_threads*4 + thread_base + LOCAL_GENERIC_START;
          translated_addrs[i] = linear_address;
      }
   } else {
      // Sub-4B access, do only one access
      assert(datasize > 0);
      num_accesses = 1;
      address_type local_word = localaddr/4;
      address_type local_word_offset = localaddr%4;
      assert( (localaddr+datasize-1)/4  == local_word ); // Make sure access doesn't overflow into next 4B chunk
      address_type linear_address = local_word*max_concurrent_threads*4 + local_word_offset + thread_base + LOCAL_GENERIC_START;
      translated_addrs[0] = linear_address;
   }
   return num_accesses;
}

/////////////////////////////////////////////////////////////////////////////////////////
int shader_core_ctx::test_res_bus(int latency){
    for(unsigned i=0; i<num_result_bus; i++){
        if(!m_result_bus[i]->test(latency)){return i;}
    }
    return -1;
}

void shader_core_ctx::execute()
{
    for(unsigned i=0; i<num_result_bus; i++){
        *(m_result_bus[i]) >>=1;
    }
    for( unsigned n=0; n < m_num_function_units; n++ ) {
        unsigned multiplier = m_fu[n]->clock_multiplier();
        for( unsigned c=0; c < multiplier; c++ ) 
            m_fu[n]->cycle();
        m_fu[n]->active_lanes_in_pipeline();
        enum pipeline_stage_name_t issue_port = m_issue_port[n];
        register_set& issue_inst = m_pipeline_reg[ issue_port ];
    warp_inst_t** ready_reg = issue_inst.get_ready();
        if( issue_inst.has_ready() && m_fu[n]->can_issue( **ready_reg ) ) {
            bool schedule_wb_now = !m_fu[n]->stallable();
            int resbus = -1;
            if( schedule_wb_now && (resbus=test_res_bus( (*ready_reg)->latency ))!=-1 ) {
                assert( (*ready_reg)->latency < MAX_ALU_LATENCY );
                m_result_bus[resbus]->set( (*ready_reg)->latency );
                m_fu[n]->issue( issue_inst );
            } else if( !schedule_wb_now ) {
                m_fu[n]->issue( issue_inst );
            } else {
                // stall issue (cannot reserve result bus)
            }
        }
    }
}

void ldst_unit::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   if( m_L1D ) {
       m_L1D->print( fp, dl1_accesses, dl1_misses );
   }
}

void ldst_unit::get_cache_stats(cache_stats &cs) {
    // Adds stats to 'cs' from each cache
    if(m_L1D)
        cs += m_L1D->get_stats();
    if(m_L1C)
        cs += m_L1C->get_stats();
    if(m_L1T)
        cs += m_L1T->get_stats();

}

void ldst_unit::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1D)
        m_L1D->get_sub_stats(css);
}
void ldst_unit::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1C)
        m_L1C->get_sub_stats(css);
}
void ldst_unit::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1T)
        m_L1T->get_sub_stats(css);
}

void shader_core_ctx::warp_inst_complete(const warp_inst_t &inst)
{
    //unsigned long long latCycles = gpu_sim_cycle - inst.get_issue_cycle();
    unsigned long long latCycles = (gpu_tot_sim_cycle + gpu_sim_cycle) - inst.get_issue_cycle();

    if (gInstrLatMap.find(inst.pc) == gInstrLatMap.end())
	{
        gInstrLatMap[inst.pc] = 0;
		gInstrNumExecMap[inst.pc] = 0;
	}
    gInstrLatMap[inst.pc] += latCycles;
	gInstrNumExecMap[inst.pc]++;
	/*
	if ((this->get_sid() == 2) && (inst.warp_id() == 39))
	{
		std::string instrStr = ptx_get_insn_str(inst.pc);
		printf("%llu: instr %s latency %llu\n", gpu_sim_cycle, instrStr.c_str(), latCycles);
	}
	*/

    if ((inst.op == LOAD_OP) || (inst.op == STORE_OP) || (inst.op == MEMORY_BARRIER_OP)) 
    {
        if (rl_scheduler::gNumWarpsExecutingMemInstr)
        {
            uint smId = this->get_sid();
            assert(gNumSMs > smId);
            rl_scheduler::gNumWarpsExecutingMemInstr[smId]--;
        }

        rl_scheduler::gNumWarpsExecutingMemInstrGPU--;

        if ((inst.space.get_type() == global_space) || 
            (inst.space.get_type() == const_space) || 
            (inst.space.get_type() == tex_space))
        {
            rl_scheduler::gNumGTCMemInstrFinished++;

            if (rl_scheduler::gGTCLongLatMemInstrCache)
            {
                //rl_scheduler::gGTCTotalLatCycles += latCycles;
                //unsigned long long avgGTCLatCycles = rl_scheduler::gGTCTotalLatCycles / rl_scheduler::gNumGTCMemInstrFinished;

                char instType[10];
                if (inst.space.get_type() == global_space)
                    strcpy(instType, "Global");
                else if (inst.space.get_type() == const_space)
                    strcpy(instType, "Const");
                else if (inst.space.get_type() == tex_space)
                    strcpy(instType, "Tex");
                else
                    strcpy(instType, "OUCH");
    
                if (latCycles >= LONG_LATENCY_MEM)
                {
                    bool add = true;
                    for (unsigned int i = 0; i < GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE; i++)
                    {
                        if (inst.pc == rl_scheduler::gGTCLongLatMemInstrCache[(m_sid * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE) + i])
                        {
                            add = false;
                            break;
                        }
                    }
                    if (add == true)
                    {
                        //if (m_sid == 0)
                            //printf("%llu: Adding inst %u(%s) as a GTC long latency instr with lat = %llu\n", gpu_sim_cycle, inst.pc, instType, latCycles);
                        rl_scheduler::gGTCLongLatMemInstrCache[(m_sid * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE) + rl_scheduler::gGTCLongLatMemInstrCacheIndex] = inst.pc;
                        rl_scheduler::gGTCLongLatMemInstrCacheIndex = (rl_scheduler::gGTCLongLatMemInstrCacheIndex + 1) % GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE;
                    }
                }
                else
                {
                    //if this pc was a long lat instr earlier, then remove it from the long lat instr cache
                    for (unsigned int i = 0; i < GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE; i++)
                    {
                        if (inst.pc == rl_scheduler::gGTCLongLatMemInstrCache[(m_sid * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE) + i])
                        {
                            rl_scheduler::gGTCLongLatMemInstrCache[(m_sid * GTC_LONG_LAT_MEM_INSTR_CACHE_SIZE) + i] = 0xdeaddead;
                            //if (m_sid == 0)
                                //printf("%llu: Removing inst %u(%s) as a GTC long latency instr with lat = %llu\n", gpu_sim_cycle, inst.pc, instType, latCycles);
                        }
                    }
                }
            }
        }
    }
    else if (rl_scheduler::gSFULongLatInstrCache && ((inst.op == SFU_OP) || (inst.op == ALU_SFU_OP)))
    {
        if (latCycles >= LONG_LATENCY)
        {
            bool add = true;
            for (unsigned int i = 0; i < SFU_LONG_LAT_INSTR_CACHE_SIZE; i++)
            {
                if (inst.pc == rl_scheduler::gSFULongLatInstrCache[(m_sid * SFU_LONG_LAT_INSTR_CACHE_SIZE) + i])
                {
                    add = false;
                    break;
                }
            }
            if (add == true)
            {
                rl_scheduler::gSFULongLatInstrCache[(m_sid * SFU_LONG_LAT_INSTR_CACHE_SIZE) + rl_scheduler::gSFULongLatInstrCacheIndex] = inst.pc;
                rl_scheduler::gSFULongLatInstrCacheIndex = (rl_scheduler::gSFULongLatInstrCacheIndex + 1) % SFU_LONG_LAT_INSTR_CACHE_SIZE;
            }
        }
        else
        {
            //if this pc was a long lat instr earlier then remove it from the long lat instr cache
            for (unsigned int i = 0; i < SFU_LONG_LAT_INSTR_CACHE_SIZE; i++)
            {
                if (inst.pc == rl_scheduler::gSFULongLatInstrCache[(m_sid * SFU_LONG_LAT_INSTR_CACHE_SIZE) + i])
                {
                    rl_scheduler::gSFULongLatInstrCache[(m_sid * SFU_LONG_LAT_INSTR_CACHE_SIZE) + i] = 0xdeaddead;
                }
            }
        }
    }

   #if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu issued@%llu\n", 
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc, gpu_tot_sim_cycle + gpu_sim_cycle, inst.get_issue_cycle()); 
   #endif
  if(inst.op_pipe==SP__OP)
      m_stats->m_num_sp_committed[m_sid]++;
  else if(inst.op_pipe==SFU__OP)
      m_stats->m_num_sfu_committed[m_sid]++;
  else if(inst.op_pipe==MEM__OP)
      m_stats->m_num_mem_committed[m_sid]++;

  if(m_config->gpgpu_clock_gated_lanes==false)
      m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
  else
      m_stats->m_num_sim_insn[m_sid] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sid]++;
  m_gpu->gpu_sim_insn += inst.active_count();
  inst.completed(gpu_tot_sim_cycle + gpu_sim_cycle);
}

void shader_core_ctx::writeback()
{

    unsigned max_committed_thread_instructions=m_config->warp_size * (m_config->pipe_widths[EX_WB]); //from the functional units
    m_stats->m_pipeline_duty_cycle[m_sid]=((float)(m_stats->m_num_sim_insn[m_sid]-m_stats->m_last_num_sim_insn[m_sid]))/max_committed_thread_instructions;

    m_stats->m_last_num_sim_insn[m_sid]=m_stats->m_num_sim_insn[m_sid];
    m_stats->m_last_num_sim_winsn[m_sid]=m_stats->m_num_sim_winsn[m_sid];

    warp_inst_t** preg = m_pipeline_reg[EX_WB].get_ready();
    warp_inst_t* pipe_reg = (preg==NULL)? NULL:*preg;
    while( preg and !pipe_reg->empty() ) {
        /*
         * Right now, the writeback stage drains all waiting instructions
         * assuming there are enough ports in the register file or the
         * conflicts are resolved at issue.
         */
        /*
         * The operand collector writeback can generally generate a stall
         * However, here, the pipelines should be un-stallable. This is
         * guaranteed because this is the first time the writeback function
         * is called after the operand collector's step function, which
         * resets the allocations. There is one case which could result in
         * the writeback function returning false (stall), which is when
         * an instruction tries to modify two registers (GPR and predicate)
         * To handle this case, we ignore the return value (thus allowing
         * no stalling).
         */
        m_operand_collector.writeback(*pipe_reg);
        unsigned warp_id = pipe_reg->warp_id();
        m_scoreboard->releaseRegisters( pipe_reg );
        m_warp[warp_id].dec_inst_in_pipeline();
        warp_inst_complete(*pipe_reg);
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
        m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
        m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
        pipe_reg->clear();
        preg = m_pipeline_reg[EX_WB].get_ready();
        pipe_reg = (preg==NULL)? NULL:*preg;
    }
}

bool ldst_unit::shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.space.get_type() != shared_space )
       return true;

   if(inst.has_dispatch_delay()){
       m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
   }

   bool stall = inst.dispatch_delay();
   if( stall ) {
       fail_type = S_MEM;
       rc_fail = BK_CONF;
   } else 
       rc_fail = NO_RC_FAIL;
   return !stall; 
}

mem_stage_stall_type
ldst_unit::process_cache_access( cache_t* cache,
                                 new_addr_type address,
                                 warp_inst_t &inst,
                                 std::list<cache_event>& events,
                                 mem_fetch *mf,
                                 enum cache_request_status status )
{
    mem_stage_stall_type result = NO_RC_FAIL;
    bool write_sent = was_write_sent(events);
    bool read_sent = was_read_sent(events);
    if( write_sent ) 
	{
        m_core->inc_store_req( inst.warp_id() );
		gStoreReqInProgress++;
		//printf("%llu: sm %u issued store for warp %u, num store reqs in progress = %u\n", gpu_sim_cycle, m_core->get_sid(), inst.warp_id(), gStoreReqInProgress);
	}
    if ( status == HIT ) {

        assert( !read_sent );
        inst.accessq_pop_back();
        if ( inst.is_load() ) {
            for ( unsigned r=0; r < 4; r++)
                if (inst.out[r] > 0)
                    m_pending_writes[inst.warp_id()][inst.out[r]]--; 
        }
        if( !write_sent ) 
            delete mf;
    } else if ( status == RESERVATION_FAIL ) {
        result = COAL_STALL;
        assert( !read_sent );
        assert( !write_sent );
        delete mf;
    } else {

        assert( status == MISS || status == HIT_RESERVED );
        //inst.clear_active( access.get_warp_mask() ); // threads in mf writeback when mf returns
        inst.accessq_pop_back();
    }
    if( !inst.accessq_empty() )
        result = BK_CONF;
    return result;
}

mem_stage_stall_type ldst_unit::process_memory_access_queue( cache_t *cache, warp_inst_t &inst )
{

    mem_stage_stall_type result = NO_RC_FAIL;
    if( inst.accessq_empty() )
        return result;


    if( !cache->data_port_free() ) 
        return DATA_PORT_STALL; 


    //const mem_access_t &access = inst.accessq_back();
    mem_fetch *mf = m_mf_allocator->alloc(inst,inst.accessq_back());
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
    return process_cache_access( cache, mf->get_addr(), inst, events, mf, status );
}

bool ldst_unit::constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.empty() || ((inst.space.get_type() != const_space) && (inst.space.get_type() != param_space_kernel)) )
       return true;
   if( inst.active_count() == 0 ) 
       return true;


   mem_stage_stall_type fail = process_memory_access_queue(m_L1C,inst);
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = C_MEM;
      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         m_stats->gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
      }
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.empty() || inst.space.get_type() != tex_space )
       return true;
   if( inst.active_count() == 0 ) 
       return true;


   mem_stage_stall_type fail = process_memory_access_queue(m_L1T,inst);
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::memory_cycle( warp_inst_t &inst, mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type )
{

   if( inst.empty() || 
       ((inst.space.get_type() != global_space) &&
        (inst.space.get_type() != local_space) &&
        (inst.space.get_type() != param_space_local)) ) 
       return true;


   if( inst.active_count() == 0 ) 
       return true;


   assert( !inst.accessq_empty() );
   mem_stage_stall_type stall_cond = NO_RC_FAIL;
   const mem_access_t &access = inst.accessq_back();

   bool bypassL1D = false; 
   if ( CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL) ) {
       bypassL1D = true; 
   } else if (inst.space.is_global()) { // global memory access 
       // skip L1 cache if the option is enabled
       if (m_core->get_config()->gmem_skip_L1D) 
           bypassL1D = true; 
   }

   if( inst.mBypassL1Cache() || bypassL1D ) {
       // bypass L1 cache
       unsigned control_size = inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
       unsigned size = access.get_size() + control_size;
       if( m_icnt->full(size, inst.is_store() || inst.isatomic()) ) {
           stall_cond = ICNT_RC_FAIL;
       } else {
           mem_fetch *mf = m_mf_allocator->alloc(inst,access);
           m_icnt->push(mf);
           inst.accessq_pop_back();
           //inst.clear_active( access.get_warp_mask() );
           if( inst.is_load() ) { 
              for( unsigned r=0; r < 4; r++) 
                  if(inst.out[r] > 0) 
                      assert( m_pending_writes[inst.warp_id()][inst.out[r]] > 0 );
           } else if( inst.is_store() ) {
              m_core->inc_store_req( inst.warp_id() );
			  gStoreReqInProgress++;
				//printf("%llu: sm %u issued store for warp %u, num store reqs in progress = %u\n", gpu_sim_cycle, m_core->get_sid(), inst.warp_id(), gStoreReqInProgress);
		   }
       }
   } else {

       assert( CACHE_UNDEFINED != inst.cache_op );
       stall_cond = process_memory_access_queue(m_L1D,inst);
   }
   if( !inst.accessq_empty() ) 
       stall_cond = COAL_STALL;
   if (stall_cond != NO_RC_FAIL) {
      stall_reason = stall_cond;
      bool iswrite = inst.is_store();
      if (inst.space.is_local()) 
         access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      else 
         access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
   }
   return inst.accessq_empty(); 
}


bool ldst_unit::response_buffer_full() const
{
    return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit::fill( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_LDST_RESPONSE_FIFO,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
}

void ldst_unit::flush(){
    // Flush L1D cache
    m_L1D->flush();
}

simd_function_unit::simd_function_unit( const shader_core_config *config )
{ 
    m_config=config;
    m_dispatch_reg = new warp_inst_t(config); 
}


sfu:: sfu(  register_set* result_port, const shader_core_config *config,shader_core_ctx *core  )
    : pipelined_simd_unit(result_port,config,config->max_sfu_latency,core)
{ 
    m_name = "SFU"; 
}

void sfu::issue( register_set& source_reg )
{
    warp_inst_t** ready_reg = source_reg.get_ready();
    //m_core->incexecstat((*ready_reg));

    (*ready_reg)->op_pipe=SFU__OP;
    m_core->incsfu_stat(m_core->get_config()->warp_size,(*ready_reg)->latency);
    pipelined_simd_unit::issue(source_reg);
}

void ldst_unit::active_lanes_in_pipeline(){
    unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
    assert(active_count<=m_core->get_config()->warp_size);
    m_core->incfumemactivelanes_stat(active_count);
}
void sp_unit::active_lanes_in_pipeline(){
    unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
    assert(active_count<=m_core->get_config()->warp_size);
    m_core->incspactivelanes_stat(active_count);
    m_core->incfuactivelanes_stat(active_count);
    m_core->incfumemactivelanes_stat(active_count);
}

void sfu::active_lanes_in_pipeline(){
    unsigned active_count=pipelined_simd_unit::get_active_lanes_in_pipeline();
    assert(active_count<=m_core->get_config()->warp_size);
    m_core->incsfuactivelanes_stat(active_count);
    m_core->incfuactivelanes_stat(active_count);
    m_core->incfumemactivelanes_stat(active_count);
}

sp_unit::sp_unit( register_set* result_port, const shader_core_config *config,shader_core_ctx *core)
    : pipelined_simd_unit(result_port,config,config->max_sp_latency,core)
{ 
    m_name = "SP "; 
}

void sp_unit :: issue(register_set& source_reg)
{
    warp_inst_t** ready_reg = source_reg.get_ready();
    //m_core->incexecstat((*ready_reg));
    (*ready_reg)->op_pipe=SP__OP;
    m_core->incsp_stat(m_core->get_config()->warp_size,(*ready_reg)->latency);
    pipelined_simd_unit::issue(source_reg);
}


pipelined_simd_unit::pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency,shader_core_ctx *core )
    : simd_function_unit(config) 
{
    m_result_port = result_port;
    m_pipeline_depth = max_latency;
    m_pipeline_reg = new warp_inst_t*[m_pipeline_depth];
    for( unsigned i=0; i < m_pipeline_depth; i++ ) 
    m_pipeline_reg[i] = new warp_inst_t( config );
    m_core=core;
}


void pipelined_simd_unit::issue( register_set& source_reg )
{
    //move_warp(m_dispatch_reg,source_reg);
    warp_inst_t** ready_reg = source_reg.get_ready();
    m_core->incexecstat((*ready_reg));
    //source_reg.move_out_to(m_dispatch_reg);
    simd_function_unit::issue(source_reg);
}

/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/

void ldst_unit::init( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc )
{
    m_memory_config = mem_config;
    m_icnt = icnt;
    m_mf_allocator=mf_allocator;
    m_core = core;
    m_operand_collector = operand_collector;
    m_scoreboard = scoreboard;
    m_stats = stats;
    m_sid = sid;
    m_tpc = tpc;
    #define STRSIZE 1024
    char L1T_name[STRSIZE];
    char L1C_name[STRSIZE];
    snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
    snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
    m_L1T = new tex_cache(L1T_name,m_config->m_L1T_config,m_sid,get_shader_texture_cache_id(),icnt,IN_L1T_MISS_QUEUE,IN_SHADER_L1T_ROB);
    m_L1C = new read_only_cache(L1C_name,m_config->m_L1C_config,m_sid,get_shader_constant_cache_id(),icnt,IN_L1C_MISS_QUEUE);
    m_L1D = NULL;
    m_mem_rc = NO_RC_FAIL;
    m_num_writeback_clients=5; // = shared memory, global/local (uncached), L1D, L1T, L1C
    m_writeback_arb = 0;
    m_next_global=NULL;
    m_last_inst_gpu_sim_cycle=0;
    m_last_inst_gpu_tot_sim_cycle=0;
}


ldst_unit::ldst_unit( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc ) : pipelined_simd_unit(NULL,config,3,core), m_next_wb(config)
{
    init( icnt,
          mf_allocator,
          core, 
          operand_collector,
          scoreboard,
          config, 
          mem_config,  
          stats, 
          sid,
          tpc );
    if( !m_config->m_L1D_config.disabled() ) {
        char L1D_name[STRSIZE];
        snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
        m_L1D = new l1_cache( L1D_name,
                              m_config->m_L1D_config,
                              m_sid,
                              get_shader_normal_cache_id(),
                              m_icnt,
                              m_mf_allocator,
                              IN_L1D_MISS_QUEUE );
    }
}

ldst_unit::ldst_unit( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats,
                      unsigned sid,
                      unsigned tpc,
                      l1_cache* new_l1d_cache )
    : pipelined_simd_unit(NULL,config,3,core), m_L1D(new_l1d_cache), m_next_wb(config)
{
    init( icnt,
          mf_allocator,
          core, 
          operand_collector,
          scoreboard,
          config, 
          mem_config,  
          stats, 
          sid,
          tpc );
}

void ldst_unit:: issue( register_set &reg_set )
{
    warp_inst_t* inst = *(reg_set.get_ready());

   // record how many pending register writes/memory accesses there are for this instruction
   assert(inst->empty() == false);
   if (inst->is_load() and inst->space.get_type() != shared_space) {
      unsigned warp_id = inst->warp_id();
      unsigned n_accesses = inst->accessq_count();
      for (unsigned r = 0; r < 4; r++) {
         unsigned reg_id = inst->out[r];
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses;
         }
      }
   }


    inst->op_pipe=MEM__OP;
    // stat collection
    m_core->mem_instruction_stats(*inst);
    m_core->incmem_stat(m_core->get_config()->warp_size,1);
    pipelined_simd_unit::issue(reg_set);
}

void ldst_unit::writeback()
{
    // process next instruction that is going to writeback
    if( !m_next_wb.empty() ) {
        if( m_operand_collector->writeback(m_next_wb) ) {
            bool insn_completed = false; 
            for( unsigned r=0; r < 4; r++ ) {
                if( m_next_wb.out[r] > 0 ) {
                    if( m_next_wb.space.get_type() != shared_space ) {
                        assert( m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0 );
                        unsigned still_pending = --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]];
                        if( !still_pending ) {
                            m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]);
                            m_scoreboard->releaseRegister( m_next_wb.warp_id(), m_next_wb.out[r] );
                            insn_completed = true; 
                        }
                    } else { // shared 
                        m_scoreboard->releaseRegister( m_next_wb.warp_id(), m_next_wb.out[r] );
                        insn_completed = true; 
                    }
                }
            }
            if( insn_completed ) {
                m_core->warp_inst_complete(m_next_wb);
            }
            m_next_wb.clear();
            m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
            m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
        }
    }

    unsigned serviced_client = -1; 
    for( unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients); c++ ) {
        unsigned next_client = (c+m_writeback_arb)%m_num_writeback_clients;
        switch( next_client ) {
        case 0: // shared memory 
            if( !m_pipeline_reg[0]->empty() ) {
                m_next_wb = *m_pipeline_reg[0];
                if(m_next_wb.isatomic()) {
                    m_next_wb.do_atomic();
                    m_core->decrement_atomic_count(m_next_wb.warp_id(), m_next_wb.active_count());
                }
                m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
                m_pipeline_reg[0]->clear();
                serviced_client = next_client; 
            }
            break;
        case 1: // texture response
            if( m_L1T->access_ready() ) {
                mem_fetch *mf = m_L1T->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 2: // const cache response
            if( m_L1C->access_ready() ) {
                mem_fetch *mf = m_L1C->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 3: // global/local
            if( m_next_global ) {
                m_next_wb = m_next_global->get_inst();
                if( m_next_global->isatomic() ) 
                    m_core->decrement_atomic_count(m_next_global->get_wid(),m_next_global->get_access_warp_mask().count());
                delete m_next_global;
                m_next_global = NULL;
                serviced_client = next_client; 
            }
            break;
        case 4: 
            if( m_L1D && m_L1D->access_ready() ) {
                mem_fetch *mf = m_L1D->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        default: abort();
        }
    }
    // update arbitration priority only if: 
    // 1. the writeback buffer was available 
    // 2. a client was serviced 
    if (serviced_client != (unsigned)-1) {
        m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients; 
    }
}

unsigned ldst_unit::clock_multiplier() const
{ 
    return m_config->mem_warp_parts; 
}
/*
void ldst_unit::issue( register_set &reg_set )
{
    warp_inst_t* inst = *(reg_set.get_ready());
   // stat collection
   m_core->mem_instruction_stats(*inst); 

   // record how many pending register writes/memory accesses there are for this instruction 
   assert(inst->empty() == false); 
   if (inst->is_load() and inst->space.get_type() != shared_space) {
      unsigned warp_id = inst->warp_id(); 
      unsigned n_accesses = inst->accessq_count(); 
      for (unsigned r = 0; r < 4; r++) {
         unsigned reg_id = inst->out[r]; 
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses; 
         }
      }
   }

   pipelined_simd_unit::issue(reg_set);
}
*/
void ldst_unit::cycle()
{
   writeback();
   m_operand_collector->step();
   for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
       if( m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage+1]->empty() )
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);

   if( !m_response_fifo.empty() ) {
       mem_fetch *mf = m_response_fifo.front();
       if (mf->istexture()) {
           if (m_L1T->fill_port_free()) {
               m_L1T->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_response_fifo.pop_front(); 
           }
       } else if (mf->isconst())  {
           if (m_L1C->fill_port_free()) {
               mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_L1C->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
               m_response_fifo.pop_front(); 
           }
       } else {
           if( mf->get_type() == WRITE_ACK || ( m_config->gpgpu_perfect_mem && mf->get_is_write() )) {
               m_core->store_ack(mf);
               m_response_fifo.pop_front();
               delete mf;
           } else {
               assert( !mf->get_is_write() ); // L1 cache is write evict, allocate line on load miss only

               bool bypassL1D = false; 
               if ( CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL) ) {
                   bypassL1D = true; 
               } else if (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == GLOBAL_ACC_W) { // global memory access 
                   if (m_core->get_config()->gmem_skip_L1D)
                       bypassL1D = true; 
               }
               if( mf->get_inst().mBypassL1Cache() || bypassL1D ) {
                   if ( m_next_global == NULL ) {
                       mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
                       m_response_fifo.pop_front();
                       m_next_global = mf;
                   }
               } else {
                   if (m_L1D->fill_port_free()) {
                       m_L1D->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                       m_response_fifo.pop_front();
                   }
               }
           }
       }
   }

   m_L1T->cycle();
   m_L1C->cycle();
   if( m_L1D ) m_L1D->cycle();

   warp_inst_t &pipe_reg = *m_dispatch_reg;
   enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
   mem_stage_access_type type;
   bool done = true;
   done &= shared_cycle(pipe_reg, rc_fail, type);
   done &= constant_cycle(pipe_reg, rc_fail, type);
   done &= texture_cycle(pipe_reg, rc_fail, type);
   done &= memory_cycle(pipe_reg, rc_fail, type);
   m_mem_rc = rc_fail;

   if (!done) { // log stall types and return
      assert(rc_fail != NO_RC_FAIL);
      m_stats->gpgpu_n_stall_shd_mem++;
      m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
      return;
   }

   if( !pipe_reg.empty() ) {
       unsigned warp_id = pipe_reg.warp_id();
       if( pipe_reg.is_load() ) {
           if( pipe_reg.space.get_type() == shared_space ) {
               if( m_pipeline_reg[2]->empty() ) {
                   // new shared memory request
                   move_warp(m_pipeline_reg[2],m_dispatch_reg);
                   m_dispatch_reg->clear();
               }
           } else {
               //if( pipe_reg.active_count() > 0 ) {
               //    if( !m_operand_collector->writeback(pipe_reg) ) 
               //        return;
               //} 

               bool pending_requests=false;
               for( unsigned r=0; r<4; r++ ) {
                   unsigned reg_id = pipe_reg.out[r];
                   if( reg_id > 0 ) {
                       if( m_pending_writes[warp_id].find(reg_id) != m_pending_writes[warp_id].end() ) {
                           if ( m_pending_writes[warp_id][reg_id] > 0 ) {
                               pending_requests=true;
                               break;
                           } else {
                               // this instruction is done already
                               m_pending_writes[warp_id].erase(reg_id); 
                           }
                       }
                   }
               }
               if( !pending_requests ) {
                   m_core->warp_inst_complete(*m_dispatch_reg);
                   m_scoreboard->releaseRegisters(m_dispatch_reg);
               }
               m_core->dec_inst_in_pipeline(warp_id);
               m_dispatch_reg->clear();
           }
       } else {
           // stores exit pipeline here
           m_core->dec_inst_in_pipeline(warp_id);
           m_core->warp_inst_complete(*m_dispatch_reg);
           m_dispatch_reg->clear();
       }
   }
}

unsigned long long gEarliestSMFinishTime = 0xFFFFFFFF;
unsigned long long gLatestSMFinishTime = 0;
extern unsigned int gHighPrioMemReq;
extern unsigned int gLowPrioMemReq;

unsigned long long gTotalNumInstrsCommitted = 0;

void printLastMemInstrInfo(const char* kernelName)
{
    std::map<std::string, unsigned int> lastMemInstrCntMap;
    for (std::map<unsigned int, std::string>::iterator iter = gLastMemInstrMap.begin(); iter != gLastMemInstrMap.end(); iter++)
    {
        unsigned int index = iter->first;
        unsigned int smId = index & 0xF;
        unsigned int dynWarpId = index >> 4;
        std::string memInstrStr = iter->second;

        if (lastMemInstrCntMap.find(memInstrStr) == lastMemInstrCntMap.end())
        {
            lastMemInstrCntMap[memInstrStr] = 1;
            printf("kernel %s sm %u dyn_warp %u last mem instr %s\n", kernelName, smId, dynWarpId, memInstrStr.c_str());
        }
        else    
            lastMemInstrCntMap[memInstrStr]++;
    }
    for (std::map<std::string, unsigned int>::iterator iter = lastMemInstrCntMap.begin(); iter != lastMemInstrCntMap.end(); iter++)
        printf("last mem inst %s executed by %u warps\n", (iter->first).c_str(), iter->second);

    gLastMemInstrMap.clear();
}

uint getPercent(float n, float d)
{
	float f = round((n * 100.0) / d);
	uint percent = f;
	return percent;
}

void shader_core_ctx::register_cta_thread_exit( unsigned cta_num )
{
   assert( m_cta_status[cta_num] > 0 );
   m_cta_status[cta_num]--;
   if (!m_cta_status[cta_num]) {
      m_n_active_cta--;
      m_barriers.deallocate_barrier(cta_num);
      shader_CTA_count_unlog(m_sid, 1);

    if (gNumWarpsAtFinishMapVec.size() != 0)
    {
        std::map<unsigned int, unsigned int>& numWarpsAtFinishMap = gNumWarpsAtFinishMapVec.at(m_sid);
        numWarpsAtFinishMap[cta_num] = 0;
    }

    if (gNumWarpsAtBarrierMapVec.size() != 0)
    {
        std::map<unsigned int, unsigned int>& numWarpsAtBarrierMap = gNumWarpsAtBarrierMapVec.at(m_sid);
        assert(numWarpsAtBarrierMap[cta_num] == 0);
    }

    if (gNumCyclesStalledAtBarrierMapVec.size() != 0)
    {
        std::map<unsigned int, unsigned long long>& numCyclesStalledAtBarrierMap = gNumCyclesStalledAtBarrierMapVec.at(m_sid);
        assert(numCyclesStalledAtBarrierMap[cta_num] == 0);
    }

    rl_scheduler::gNumFinishedTBs++;

    unsigned index = m_sid * MAX_NUM_TB_PER_SM + cta_num;
    unsigned int numInstrExeced = 0;

    if (gTBProgressArray)
    {
        bool printFlag = true;
        if (printFlag && (rl_scheduler::gNumFinishedTBs == 1))
        {
            unsigned int maxProgress = 0;
            for (unsigned int i = 0; i < MAX_NUM_TB_PER_SM; i++)
            {
                unsigned idx = m_sid * MAX_NUM_TB_PER_SM + i;
                unsigned int tbProgress = gTBProgressArray[idx];
                if ((tbProgress != 0) && (tbProgress != 0xdeaddead))
                    if (maxProgress < tbProgress)
                        maxProgress = tbProgress;
            }
            for (unsigned int i = 0; i < MAX_NUM_TB_PER_SM; i++)
            {
                unsigned idx = m_sid * MAX_NUM_TB_PER_SM + i;
                unsigned int tbProgress = gTBProgressArray[idx];
                if ((tbProgress != 0) && (tbProgress != 0xdeaddead))
                    printf("%llu: tb %u progress %u(%f)\n", gpu_sim_cycle, i, tbProgress/32, (float)tbProgress/(float)maxProgress);
            }
        }

        rl_scheduler::gNumInstrsExecedByFinishedTBs += gTBProgressArray[index];
        numInstrExeced = gTBProgressArray[index];

        if (gTBProgressArray[index] > gMaxNumInstrsExecedByTB)
            gMaxNumInstrsExecedByTB = gTBProgressArray[index];

        if (gTBProgressArray[index] < gMinNumInstrsExecedByTB)
            gMinNumInstrsExecedByTB = gTBProgressArray[index];

        gTBProgressArray[index] = 0;
    }

    if (rl_scheduler::gTBNumSpInstrsArray)
        rl_scheduler::gTBNumSpInstrsArray[index] = 0;
    if (rl_scheduler::gTBNumSfuInstrsArray)
        rl_scheduler::gTBNumSfuInstrsArray[index] = 0;
    if (rl_scheduler::gTBNumMemInstrsArray)
        rl_scheduler::gTBNumMemInstrsArray[index] = 0;

    if (gSelectedTB && (gSelectedTB[m_sid] == cta_num))
        gSelectedTB[m_sid] = 0xdeaddead;

    scheduler_unit* sched = schedulers[0];
    assert(sched);

	rl_scheduler* lRLSched = 0;
	if (schedulers[0]->isRLSched())
		lRLSched = (rl_scheduler*)sched;

    printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld), executed %u instructions, %u CTAs running, total %u(%u) finished by all SMs\n", m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle, numInstrExeced/32, m_n_active_cta, rl_scheduler::gNumFinishedTBs, gTotalNumOfTBsInGrid );

	unsigned int lNumTBsResident = gNumTBsPerSM * gNumSMs;
    bool moreTBsLeft = m_cluster->get_gpu()->get_more_cta_left() ? true : false;

	if (lRLSched)
	{
		if (gTwoPhase == 0)
		{
			unsigned int lMaxNumTBsInPhase = gTotalNumOfTBsInGrid;
			unsigned int lNumTBsRemainingInPhase = lMaxNumTBsInPhase - rl_scheduler::gNumFinishedTBs;
			if ((gStnryFlag == 0) /*|| gUseWhichSchedAsAction*/)
			{
				//keep the orig alpha expl values
				lRLSched->dCurrExplorationPercent = lRLSched->dOrigExplorationPercent;
				lRLSched->dCurrAlpha = lRLSched->dOrigAlpha;
			}
			else
			{
				//gradually reduce the orig alpha expl values
				lRLSched->dCurrExplorationPercent = (lRLSched->dOrigExplorationPercent * lNumTBsRemainingInPhase) / lMaxNumTBsInPhase;
				lRLSched->dCurrAlpha = lRLSched->dOrigAlpha * ((float)lNumTBsRemainingInPhase / (float)lMaxNumTBsInPhase);
				printf("1. %llu curr alpha = %f, curr expl percent = %f\n", gpu_sim_cycle, lRLSched->dCurrAlpha, lRLSched->dCurrExplorationPercent);
			}
		}
		else
		{
			unsigned int lMaxNumTBsInPhase1 = gTotalNumOfTBsInGrid - lNumTBsResident;
			if (moreTBsLeft && (lMaxNumTBsInPhase1 > rl_scheduler::gNumFinishedTBs))
			{
				unsigned int lNumTBsRemainingInPhase1 = lMaxNumTBsInPhase1 - rl_scheduler::gNumFinishedTBs;
				if ((gStnryFlag == 0) /*|| gUseWhichSchedAsAction*/)
				{
					//keep the orig alpha expl values
					lRLSched->dCurrExplorationPercent = lRLSched->dOrigExplorationPercent;
					lRLSched->dCurrAlpha = lRLSched->dOrigAlpha;
				}
				else
				{
					//gradually reduce the orig alpha expl values
					lRLSched->dCurrExplorationPercent = (lRLSched->dOrigExplorationPercent * lNumTBsRemainingInPhase1) / lMaxNumTBsInPhase1;
					lRLSched->dCurrAlpha = lRLSched->dOrigAlpha * ((float)lNumTBsRemainingInPhase1 / (float)lMaxNumTBsInPhase1);
					printf("1. %llu curr alpha = %f, curr expl percent = %f\n", gpu_sim_cycle, lRLSched->dCurrAlpha, lRLSched->dCurrExplorationPercent);
				}
			}
			else
			{
				//keep the orig alpha expl values
				lRLSched->dCurrAlpha = lRLSched->dOrigAlpha;
				lRLSched->dCurrExplorationPercent = lRLSched->dOrigExplorationPercent;
				printf("2. curr alpha = %f, curr expl percent = %f\n", lRLSched->dCurrAlpha, lRLSched->dOrigExplorationPercent);
			}
		}
	}

    printf("Shader %d CTA #%d, min warp time %u, max warp time %u, warp divergence = %f\n", m_sid, cta_num, gTBMinWarpTimeArray[index], gTBMaxWarpTimeArray[index], ((float)gTBMaxWarpTimeArray[index]) / ((float)gTBMinWarpTimeArray[index]));
	gTBMinWarpTimeArray[index] = 0xFFFFFFFF;
	gTBMaxWarpTimeArray[index] = 0;

	if( m_n_active_cta == 0 ) {
        assert( m_kernel != NULL );
        m_kernel->dec_running();
        printf("GPGPU-Sim uArch: Shader %u empty (release kernel %u \'%s\').\n", m_sid, m_kernel->get_uid(),
                 m_kernel->name().c_str() );

  		unsigned long long lTotalNumInstrsCommittedBySM = m_stats->m_num_sp_committed[m_sid] + m_stats->m_num_sfu_committed[m_sid] + m_stats->m_num_mem_committed[m_sid];
		gTotalNumInstrsCommitted += lTotalNumInstrsCommittedBySM;

		gModifiedAttrCombNumInstrVec[m_sid % gNumSmGroups] += lTotalNumInstrsCommittedBySM;
		if (gpu_sim_cycle > gModifiedAttrCombMaxSimCyclesVec[m_sid % gNumSmGroups])
			gModifiedAttrCombMaxSimCyclesVec[m_sid % gNumSmGroups] = gpu_sim_cycle;

/*
		if (gPrintQvaluesFlag)
		{
			gPrintQvaluesFlag = false;
        	scheduler_unit* sched = schedulers[0];
        	sched->printQvalues();
		}
*/

		if (gEarliestSMFinishTime > gpu_sim_cycle)
            gEarliestSMFinishTime = gpu_sim_cycle;
		if (gLatestSMFinishTime < gpu_sim_cycle)
            gLatestSMFinishTime = gpu_sim_cycle;

        if( m_kernel->no_more_ctas_to_run() ) {
          	if( !m_kernel->running() ) {
				numKernelsFinished++;

            	printf("GPGPU-Sim uArch: GPU detected kernel \'%s\' finished on shader %u.\n", m_kernel->name().c_str(), m_sid );
				printf("Max warp drain time = %u, min warp drain time = %u, avg warp drain time = %u\n", gMaxDrainTime, gMinDrainTime, gTotalDrainTime / gTotalNumWarpsFinished);
				gPrintFlag = true;
				printf("gCnt0 = %u, gCnt1 = %u, gCnt1_5 = %u, gCnt2 = %u, gCnt2_1 = %u, gCnt2_2 = %u, gCnt2_3=%u, gCnt2_4=%u, gCnt2_5=%u, gCnt3 = %u, gCnt4 =%u, gCnt5=%u, gCnt6=%u, gCnt7=%u, gCnt8=%u\n", gCnt0, gCnt1, gCnt1_5, gCnt2, gCnt2_1, gCnt2_2, gCnt2_3, gCnt2_4, gCnt2_5, gCnt3, gCnt4, gCnt5, gCnt6, gCnt7, gCnt8);
				gCnt0 = 0;
				gCnt1 = 0;
				gCnt1_5 = 0;
				gCnt2 = 0;
				gCnt2_1 = 0;
				gCnt2_2 = 0;
				gCnt2_3 = 0;
				gCnt2_4 = 0;
				gCnt2_5 = 0;
				gCnt3 = 0;
				gCnt4 = 0;
				gCnt5 = 0;
				gCnt6 = 0;
				gCnt7 = 0;
				gCnt8 = 0;
				printf ("Num Ready Mem Instr Warps %u, Num Ready Alu Instr Warps %u, Num Warps waiting for data %u\n", gNumReadyMemInstrWarps, gNumReadyAluInstrWarps, gNumWarpsWaitingForData);
				gNumReadyMemInstrWarps = 0;
 				gNumReadyAluInstrWarps = 0;
				gNumWarpsWaitingForData = 0;

				if (gPrintQvaluesFlag)
				{
					//gPrintQvaluesFlag = false;
        			scheduler_unit* sched = schedulers[0];
        			sched->printQvalues();
				}

                if (rl_scheduler::gNumFinishedTBs != 0)
                {
                      unsigned int avgNumInstrs = rl_scheduler::gNumInstrsExecedByFinishedTBs/rl_scheduler::gNumFinishedTBs;
                      printf("avg num instrs by a tb = %u, max = %u, min = %u, max/avg = %f, max/min = %f, avg/min = %f\n", 
                               avgNumInstrs / 32, gMaxNumInstrsExecedByTB / 32, gMinNumInstrsExecedByTB / 32, 
                             (float)gMaxNumInstrsExecedByTB / (float)avgNumInstrs,
                             (float)gMaxNumInstrsExecedByTB / (float)gMinNumInstrsExecedByTB,
                             (float)avgNumInstrs / (float)gMinNumInstrsExecedByTB);
                }

                printLastMemInstrInfo(m_kernel->name().c_str());
                printf("gEarliestSMFinishTime = %llu, gLatestSMFinishTime = %llu, %f\n", gEarliestSMFinishTime, gLatestSMFinishTime, ((float)gLatestSMFinishTime/(float)gEarliestSMFinishTime));
                gEarliestSMFinishTime = 0xFFFFFFFF;
                gLatestSMFinishTime = 0;
                gPrintNoMoreCTAsMsg = true;

                printf("high prio mem req = %u, low prio mem req = %u\n", gHighPrioMemReq, gLowPrioMemReq);
                gHighPrioMemReq = 0;
                gLowPrioMemReq = 0;

		  		if ((gGtoSchedCnt + gLrrSchedCnt) > 0)
		  			printf("Num times LRR sched used %u(%f), Num times GTO sched used %u(%f)\n", gLrrSchedCnt, (float)gLrrSchedCnt/(float)(gLrrSchedCnt+gGtoSchedCnt), gGtoSchedCnt, (float)gGtoSchedCnt/(float)(gLrrSchedCnt+gGtoSchedCnt));
                m_gpu->set_kernel_done( m_kernel );

                rl_scheduler::gNumFinishedTBs = 0;
                rl_scheduler::gNumInstrsExecedByFinishedTBs = 0;

				gNumCellsTouchedPrimary = 0;
				gNumCellsTouchedSecondary = 0;

				rl_scheduler* schedPtr = ((rl_scheduler*)schedulers[0]);
				if (schedPtr->isRLSched())
				{
					printf("Primary Action Info\n");
					unsigned long long totalActions = 0;
					for (std::map<unsigned int, unsigned int>::iterator iter = gPrimaryActionCntMap.begin();
					 	iter != gPrimaryActionCntMap.end();
					 	iter++)
					{
						totalActions += (iter->second);
					}
					for (std::map<unsigned int, unsigned int>::iterator iter = gPrimaryActionCntMap.begin();
					 	iter != gPrimaryActionCntMap.end();
					 	iter++)
					{
						std::string actionStr = schedPtr->dRLEngines[0]->getActionStr(iter->first);
						printf("Action %u(%s) number of times selected %u(%llu percent)\n", iter->first, actionStr.c_str(), iter->second, 
						        ((unsigned long long)(iter->second) * 100) / totalActions);
						iter->second = 0;
					}
					gPrimaryActionCntMap.clear();
					schedPtr->dRLEngines[0]->printPrimaryActionCntSnapshots();
					gPrimaryActionCntSnapshotCycle = 0;
					printf("\n");
	
					printf("Secondary Action Info\n");
					for (std::map<unsigned int, unsigned int>::iterator iter = gSecondaryActionCntMap.begin();
					 	iter != gSecondaryActionCntMap.end();
					 	iter++)
					{
						printf("Action %u number of times selected %u\n", iter->first, iter->second);
						iter->second = 0;
					}
					gSecondaryActionCntMap.clear();
					if ((gPrimaryFirstCnt + gSecondaryFirstCnt) > 0)
					{
						printf("Primary first action = %u(%f), Secondary First Action = %u(%f)\n", gPrimaryFirstCnt, 
					   			(float)gPrimaryFirstCnt / (float)(gPrimaryFirstCnt + gSecondaryFirstCnt), 
								gSecondaryFirstCnt, (float)gSecondaryFirstCnt / (float)(gPrimaryFirstCnt + gSecondaryFirstCnt));
					}
				}

				for (unsigned int i = 0; i < gNumSmGroups; i++)
				{
					double lAvgIPC = (float)gModifiedAttrCombNumInstrVec[i] / ((float)gModifiedAttrCombMaxSimCyclesVec[i] * gNumSmsPerGroup);
					printf("Avg IPC of attr comb %s is %f(%llu/%llu)\n", gModifiedAttrCombStrVec[i].c_str(), lAvgIPC, gModifiedAttrCombNumInstrVec[i], gModifiedAttrCombMaxSimCyclesVec[i]);
					unsigned long long lSimCycles = (float)gTotalNumInstrsCommitted / (gNumSmGroups * gNumSmsPerGroup * (float)lAvgIPC);
					printf("simulation cycles of %s = %llu for %llu instrs\n", gModifiedAttrCombStrVec[i].c_str(), lSimCycles, gTotalNumInstrsCommitted);
					printf("ATTRCOMB %s KERNEL %s CYCLES %llu\n", gModifiedAttrCombStrVec[i].c_str(), m_kernel->name().c_str(), lSimCycles);
				}

				gTotalNumInstrsCommitted = 0;
				for (unsigned int i = 0; i < gNumSmGroups; i++)
				{
					gModifiedAttrCombNumInstrVec[i] = 0;
					gModifiedAttrCombMaxSimCyclesVec[i] = 0;
				}

				bool printLatencyInfo = false;
				if (printLatencyInfo)
				{
					float totalAvgLat = 0.0;
					for (std::map<unsigned int, unsigned long long>::iterator iter1 = gInstrLatMap.begin(); iter1 != gInstrLatMap.end(); iter1++)
					{
						unsigned int pc = iter1->first;
						unsigned long long totalLat = iter1->second;
						unsigned long long numExecs = gInstrNumExecMap[pc];
	
						float avgLat = (double)totalLat/(double)numExecs;
						printf("instr pc %u (%s), total latency = %llu, num execs = %llu, avg latency = %f\n", pc, ptx_get_insn_str(pc).c_str(), totalLat, numExecs, (double)totalLat/(double)numExecs);
						totalAvgLat += avgLat;
					}
					printf("total avg lat = %f\n", totalAvgLat);
				}

				printf("total gpu wide rewards = %d\n", gTotalGPUWideReward);
				printf("total gpu wide discounted rewards = %f\n", gTotalGPUWideDiscountedReward);

				schedPtr = ((rl_scheduler*)schedulers[0]);
				if (schedPtr->isRLSched())
				{
					unsigned int lNumCellsTouched;
					unsigned int lNumUpdates = schedPtr->dRLEngines[0]->mGetTotalQvalueUpdates(lNumCellsTouched);
					printf("Primary: total q value = %e total num of updates = %u num of cells touched = %u\n", schedPtr->dRLEngines[0]->mGetTotalQvalue(), lNumUpdates,lNumCellsTouched);
					if (schedPtr->dRLEngines[1])
					{
						lNumUpdates = schedPtr->dRLEngines[1]->mGetTotalQvalueUpdates(lNumCellsTouched);
						printf("Secondary: total q value = %e total num of updates = %u num of cells touched = %u\n", schedPtr->dRLEngines[1]->mGetTotalQvalue(), lNumUpdates, lNumCellsTouched);
					}
				}
				unsigned int total = gSameWarpAsGTOCnt + gNotSameWarpAsGTOCnt + 1; //adding 1 to make it non zero
				printf("RLWS action matched GTO action %u (%f) times, did not match %u (%f) times\n", gSameWarpAsGTOCnt, (float)gSameWarpAsGTOCnt/(float)total, gNotSameWarpAsGTOCnt, (float)gNotSameWarpAsGTOCnt/(float)total);
				for (std::map<unsigned int, unsigned int>::iterator iter = gPossiblePrimaryActionCntMap.begin(); iter != gPossiblePrimaryActionCntMap.end(); iter++)
					if (iter->second > 0)
						printf("num of times %u possible actions = %u\n", iter->first, iter->second);
				for (unsigned int i = 0; i < 50; i++)
					gPossiblePrimaryActionCntMap[i] = 0;
				gSameWarpAsGTOCnt = 0;
				gNotSameWarpAsGTOCnt = 0;

				bool gPrintRunStallCycles = false;
				if (gPrintRunStallCycles)
				{
					printf("Stall Run Info:\n");
					scheduler_unit* schedPtr;
					if ((schedulers[0]->lastSchedRunCycle) > (schedulers[1]->lastSchedRunCycle))
						schedPtr = schedulers[0];
					else
						schedPtr = schedulers[1];

					std::map<unsigned int, unsigned int> stallCycleMap;
					std::map<unsigned int, unsigned int> runCycleMap;
					unsigned int numBuckets = 6;
					for (uint i = 0; i < numBuckets; i++)
					{
						if (numKernelsFinished == 1)
						{
							gBmStallCycleMap[i] = 0;
							gBmRunCycleMap[i] = 0;
						}
						stallCycleMap[i] = 0;
						runCycleMap[i] = 0;
					}
					std::map<unsigned int, unsigned int> runStallBucketStartMap;
					runStallBucketStartMap[0] = 0;
					runStallBucketStartMap[1] = 10;
					runStallBucketStartMap[2] = 25;
					runStallBucketStartMap[3] = 75;
					runStallBucketStartMap[4] = 150;
					runStallBucketStartMap[5] = 300;
					runStallBucketStartMap[6] = 10000;
						
					schedPtr->runStallCyclesVec.push_back(schedPtr->consCycleCnt); //push the last cycle cnt
					for (uint i = 0; i < schedPtr->runStallCyclesVec.size(); i++)
					{
						unsigned int numCycles = schedPtr->runStallCyclesVec[i];
						unsigned int bucket = 0;
						if (numCycles >= 300)
							bucket = 5;
						else if (numCycles >= 150)
							bucket = 4;
						else if (numCycles >= 75)
							bucket = 3;
						else if (numCycles >= 25)
							bucket = 2;
						else if (numCycles >= 10)
							bucket = 1;
						else
							bucket = 0;

						assert(bucket < numBuckets);

						if ((i & 0x1) == 0)
						{
							if (schedPtr->firstCycleRunCycle)
								runCycleMap[bucket] += numCycles;
							else
								stallCycleMap[bucket] += numCycles;
						}
						else
						{
							if (schedPtr->firstCycleRunCycle)
								stallCycleMap[bucket] += numCycles;
							else
								runCycleMap[bucket] += numCycles;
						}
					}

					//generate csv for sched state
					printf("schedState\n");
					for (uint i = 0; i < schedPtr->schedStateVec.size(); i++)
					{
						printf("%u\n", schedPtr->schedStateVec[i]);
					}
					printf("DONE schedState\n");
					//generate run stall csv for this scheduler
					printf("RUN STALL\n");
					bool isRun = true;
					if (schedPtr->firstCycleRunCycle == false)
					{
						printf("0");
						isRun = false;
					}
					for (uint i = 0; i < schedPtr->runStallCyclesVec.size(); i++)
					{
						unsigned int numCycles = schedPtr->runStallCyclesVec[i];
						if (isRun == true)
						{
							printf("%u", numCycles);
							isRun = false;
						}
						else
						{
							printf(" %u\n", numCycles);
							isRun = true;
						}
					}
					printf("DONE RUN STALL\n");

					schedPtr->runStallCyclesVec.clear();
					schedPtr->schedStateVec.clear();
					schedPtr->lastSchedRunCycle = 0;

					if (gKernelRunCycleMapMap.find(m_kernel->name()) == gKernelRunCycleMapMap.end())
					{
						std::map<unsigned int, unsigned int> m1;
						gKernelRunCycleMapMap[m_kernel->name()] = m1;

						std::map<unsigned int, unsigned int> m2;
						gKernelStallCycleMapMap[m_kernel->name()] = m2;
					}

					std::map<unsigned int, unsigned int>& kernelRunCycleMap = gKernelRunCycleMapMap[m_kernel->name()];
					std::map<unsigned int, unsigned int>& kernelStallCycleMap = gKernelStallCycleMapMap[m_kernel->name()];

					unsigned int totalRunCycles = 0;
					unsigned int totalKernelRunCycles = 0;
					for (std::map<unsigned int, unsigned int>::iterator iter = runCycleMap.begin();
					     iter != runCycleMap.end();
						 iter++)
					{
						//printf("bucket %u (%u-%u) num run cycles %u\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second);
						totalRunCycles += iter->second;
						gBmRunCycleMap[iter->first] += iter->second;
						kernelRunCycleMap[iter->first] += iter->second;
						totalKernelRunCycles += kernelRunCycleMap[iter->first];
					}

					unsigned int totalStallCycles = 0;
					unsigned int totalKernelStallCycles = 0;
					for (std::map<unsigned int, unsigned int>::iterator iter = stallCycleMap.begin();
					     iter != stallCycleMap.end();
						 iter++)
					{
						//printf("bucket %u (%u-%u) num stall cycles %u\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second);
						totalStallCycles += iter->second;
						gBmStallCycleMap[iter->first] += iter->second;
						kernelStallCycleMap[iter->first] += iter->second;
						totalKernelStallCycles += kernelStallCycleMap[iter->first];
					}
					char csvString[1024];
					printf("For kernel %s\n", m_kernel->name().c_str());
					unsigned int totalCycles = totalRunCycles + totalStallCycles;
					printf("total run cycles of this kernel call = %u(%f)\n", totalRunCycles, (float)totalRunCycles/(float)totalCycles);
					printf("total stall cycles of this kernel call = %u(%f)\n", totalStallCycles, (float)totalStallCycles/(float)totalCycles);

					unsigned int totalKernelCycles = totalKernelRunCycles + totalKernelStallCycles;
					printf("total kernel run cycles = %u(%f)\n", totalKernelRunCycles, (float)totalKernelRunCycles/(float)totalKernelCycles);
					printf("total kernel stall cycles = %u(%f)\n", totalKernelStallCycles, (float)totalKernelStallCycles/(float)totalKernelCycles);
					char bmNameStr[1024];
					strcpy(bmNameStr, get_current_dir_name());
					char* bmNamePtr = strrchr(bmNameStr, '/');
					bmNamePtr++;

					sprintf(csvString, "CSV: %s %u %u", bmNamePtr, getPercent(totalKernelRunCycles, totalKernelCycles), getPercent(totalKernelStallCycles, totalKernelCycles));
					for (std::map<unsigned int, unsigned int>::iterator iter = kernelRunCycleMap.begin();
				     	iter != kernelRunCycleMap.end();
					 	iter++)
					{
						printf("bucket %u (%u-%u) num run cycles %u(%f)\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second, (float)iter->second/(float)totalKernelRunCycles);
						sprintf(csvString, "%s %u", csvString, getPercent(iter->second, totalKernelRunCycles));
					}
					printf("\n");
					for (std::map<unsigned int, unsigned int>::iterator iter = kernelStallCycleMap.begin();
				     	iter != kernelStallCycleMap.end();
					 	iter++)
					{
						printf("bucket %u (%u-%u) num stall cycles %u(%f)\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second, (float)iter->second/(float)totalKernelStallCycles);
						sprintf(csvString, "%s %u", csvString, getPercent(iter->second, totalKernelStallCycles));
					}
					printf("%s\n", csvString);

					if (numKernelsFinished > 1)
					{
						printf("At the end of %u kernels:\n", numKernelsFinished);
						unsigned int totalBMRunCycles = 0;
						unsigned int totalBMStallCycles = 0;


						for (std::map<unsigned int, unsigned int>::iterator iter = gBmRunCycleMap.begin();
				     		iter != gBmRunCycleMap.end();
					 		iter++)
						{
							totalBMRunCycles += iter->second;
						}
						for (std::map<unsigned int, unsigned int>::iterator iter = gBmStallCycleMap.begin();
				     		iter != gBmStallCycleMap.end();
					 		iter++)
						{
							totalBMStallCycles += iter->second;
						}
						unsigned int totalBMCycles = totalBMRunCycles + totalBMStallCycles;
						printf("total bm run cycles = %u(%f)\n", totalBMRunCycles, (float)totalBMRunCycles/(float)totalBMCycles);
						printf("total bm stall cycles = %u(%f)\n", totalBMStallCycles, (float)totalBMStallCycles/(float)totalBMCycles);
						sprintf(csvString, "CSV:%s %u %u", bmNamePtr, getPercent(totalBMRunCycles, totalBMCycles), getPercent(totalBMStallCycles, totalBMCycles));

						for (std::map<unsigned int, unsigned int>::iterator iter = gBmRunCycleMap.begin();
				     		iter != gBmRunCycleMap.end();
					 		iter++)
						{
							printf("bucket %u (%u-%u) num run cycles %u(%f)\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second, (float)iter->second/(float)totalBMRunCycles);
							sprintf(csvString, "%s %u", csvString, getPercent(iter->second, totalBMRunCycles));
						}

						for (std::map<unsigned int, unsigned int>::iterator iter = gBmStallCycleMap.begin();
				     		iter != gBmStallCycleMap.end();
					 		iter++)
						{
							printf("bucket %u (%u-%u) num stall cycles %u(%f)\n", iter->first, runStallBucketStartMap[iter->first], runStallBucketStartMap[iter->first + 1], iter->second, (float)iter->second/(float)totalBMStallCycles);
							sprintf(csvString, "%s %u", csvString, getPercent(iter->second, totalBMStallCycles));
						}
						printf("%s\n", csvString);
					}

					printf("run cycle map info:\n");
					printf("cycles freq\n");
					for (std::map<unsigned int, unsigned int>::iterator iter = schedPtr->runCycleCntMap.begin();
						 iter != schedPtr->runCycleCntMap.end();
						 iter++)
					{
						printf("%u %u\n", iter->first, iter->second);
					}
					printf("done run cycle map info:\n");

					printf("stall cycle map info:\n");
					printf("cycles freq\n");
					for (std::map<unsigned int, unsigned int>::iterator iter = schedPtr->stallCycleCntMap.begin();
						 iter != schedPtr->stallCycleCntMap.end();
						 iter++)
					{
						printf("%u %u\n", iter->first, iter->second);
					}
					printf("done stall cycle map info:\n");
				}

				if (gPrintAttrValueCnts)
				{
					printf("attr val cnt info \n");
					for (std::map<std::string, std::map<unsigned int, unsigned int> >::iterator iter1 =  gAttrNameValueCntMap.begin();
						 iter1 != gAttrNameValueCntMap.end();
						 iter1++)
					{
						std::string attrName = iter1->first;
						std::map<unsigned int, unsigned int>& valCntMap = iter1->second;
						for (std::map<unsigned int, unsigned int>::iterator iter2 = valCntMap.begin();
							 iter2 != valCntMap.end();
							 iter2++)
						{
							unsigned int val = iter2->second;
							//uint percent = f;
							unsigned int avgVal = (val / (gNumSMs * NUM_SCHED_PER_SM)) + ((val % (gNumSMs * NUM_SCHED_PER_SM)) ? 1 : 0);
							printf("attr %s value %u cnt %u avg per sched %u\n", attrName.c_str(), iter2->first, val, avgVal);
						}
					}
					gAttrNameValueCntMap.clear();
				}
/*
				bool printTimeWarpIdSchedInfo = false;
				if (printTimeWarpIdSchedInfo)
				{
					printf("time warp id sched info:\n");
					for (std::map<unsigned long long, unsigned int>::iterator iter =  gCycleWarpMap.begin();
				     	iter != gCycleWarpMap.end();
					 	iter++)
					{
						printf("%llu %u\n", iter->first, iter->second);
					}
					printf("done time warp id sched info:\n");
				}
*/

				for (unsigned int id = 0; id < gNumSMs; id++)
				{
    				std::map<unsigned int, unsigned int> splitWarpDynamicIdMap;
					gSplitWarpDynamicIdMapVec[id] = splitWarpDynamicIdMap;
				}

				bool printStateActionWtArr = true;
				if (rl_scheduler::gUseFeatureWeightFuncApprox && printStateActionWtArr && (schedPtr->isRLSched()))
				{
					printf("Printing state action weight array:\n");
					for (unsigned int i = 0; i < (schedPtr->dRLEngines[0]->dAttributeVector.size() * schedPtr->dRLEngines[0]->dNumActions); i++)
					{
						printf("%e\n", schedPtr->dRLEngines[0]->dStateActionWeightArray[i]);
					}
					printf("DONE printing state action weight array:\n");
				
					printWeights();
				}

				if ((rl_scheduler::gUseFeatureWeightFuncApprox == false) && schedPtr->isRLSched())
				{
					printQvalueSamples();
					printQvalueUpdateSamples();
				}

				if (gShareQvalueTableForAllSMs && (schedPtr->isRLSched()))
				{
					schedPtr->dRLEngines[0]->mClear();
				}
				gBmName = 0;
				gKernelName = 0;

				if (gRTOSched)
				{
					unsigned int totalCnt = gRTOSchedRandomOrderCnt + gRTOSchedGTOOrderCnt;
					printf("RTO Sched Random order cnt = %u(%f), GTO order cnt = %u(%f)\n", gRTOSchedRandomOrderCnt, 
						   (float)gRTOSchedRandomOrderCnt/(float)totalCnt, gRTOSchedGTOOrderCnt, 
						   (float)gRTOSchedGTOOrderCnt/(float)totalCnt);
				}
				if (gTmpStateActionFeatureValueArray)
				{
					delete gTmpStateActionFeatureValueArray;
					gTmpStateActionFeatureValueArray = 0;
				}
				gTotalNumWarpsFinished = 0;
				gIPAWS_UseGTO = false;
			}
		}

        for (unsigned int j = 0; j < schedulers.size(); j++)
        {
            scheduler_unit* sched = schedulers[j];
            sched->clear();
			if (sched->isRLSched())
			{
				rl_scheduler* rlSched = (rl_scheduler*)sched;
				rlSched->dCurrAlpha = rlSched->dOrigAlpha;
				rlSched->dCurrExplorationPercent = rlSched->dOrigExplorationPercent;
			}
        }
        m_kernel=NULL;
        fflush(stdout);
      }
   }
}

void gpgpu_sim::shader_print_runtime_stat( FILE *fout ) 
{
    /*
   fprintf(fout, "SHD_INSN: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_not_completed());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
   fprintf(fout, "\n");
   */
}


void gpgpu_sim::shader_print_scheduler_stat( FILE* fout, bool print_dynamic_info ) const
{
    // Print out the stats from the sampling shader core
    const unsigned scheduler_sampling_core = m_shader_config->gpgpu_warp_issue_shader;
    #define STR_SIZE 55
    char name_buff[ STR_SIZE ];
    name_buff[ STR_SIZE - 1 ] = '\0';
    const std::vector< unsigned >& distro
        = print_dynamic_info ?
          m_shader_stats->get_dynamic_warp_issue()[ scheduler_sampling_core ] :
          m_shader_stats->get_warp_slot_issue()[ scheduler_sampling_core ];
    if ( print_dynamic_info ) {
        snprintf( name_buff, STR_SIZE - 1, "dynamic_warp_id" );
    } else {
        snprintf( name_buff, STR_SIZE - 1, "warp_id" );
    }
    fprintf( fout,
             "Shader %d %s issue ditsribution:\n",
             scheduler_sampling_core,
             name_buff );
    const unsigned num_warp_ids = distro.size();
    // First print out the warp ids
    fprintf( fout, "%s:\n", name_buff );
    for ( unsigned warp_id = 0;
          warp_id < num_warp_ids;
          ++warp_id  ) {
        fprintf( fout, "%d, ", warp_id );
    }

    fprintf( fout, "\ndistro:\n" );
    // Then print out the distribution of instuctions issued
    for ( std::vector< unsigned >::const_iterator iter = distro.begin();
          iter != distro.end();
          iter++ ) {
        fprintf( fout, "%d, ", *iter );
    }
    fprintf( fout, "\n" );
}

void gpgpu_sim::shader_print_cache_stats( FILE *fout ) const{

    // L1I
    struct cache_sub_stats total_css;
    struct cache_sub_stats css;

    if(!m_shader_config->m_L1I_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "\n========= Core cache stats =========\n");
        fprintf(fout, "L1I_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1I_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1I_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1I_total_cache_misses = %u\n", total_css.misses);
        fprintf(fout, "\tL1I_total_cache_cold_misses = %u\n", total_css.cold_misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
            fprintf(fout, "\tL1I_total_cache_cold_miss_rate = %.4lf\n", (double)total_css.cold_misses / (double)total_css.misses);
        }
        fprintf(fout, "\tL1I_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1I_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }

    // L1D
    if(!m_shader_config->m_L1D_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1D_cache:\n");
        for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++){
            m_cluster[i]->get_L1D_sub_stats(css);

            fprintf( stdout, "\tL1D_cache_core[%d]: Access = %d, Miss = %d, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                     i, css.accesses, css.misses, (double)css.misses / (double)css.accesses, css.pending_hits, css.res_fails);

            total_css += css;
        }
        fprintf(fout, "\tL1D_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1D_total_cache_misses = %u\n", total_css.misses);
        fprintf(fout, "\tL1D_total_cache_cold_misses = %u\n", total_css.cold_misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
            fprintf(fout, "\tL1D_total_cache_cold_miss_rate = %.4lf\n", (double)total_css.cold_misses / (double)total_css.misses);
        }
        fprintf(fout, "\tL1D_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1D_total_cache_reservation_fails = %u\n", total_css.res_fails);
        total_css.print_port_stats(fout, "\tL1D_cache"); 
    }

    // L1C
    if(!m_shader_config->m_L1C_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1C_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1C_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1C_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1C_total_cache_misses = %u\n", total_css.misses);
        fprintf(fout, "\tL1C_total_cache_cold_misses = %u\n", total_css.cold_misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
            fprintf(fout, "\tL1C_total_cache_cold_miss_rate = %.4lf\n", (double)total_css.cold_misses / (double)total_css.misses);
        }
        fprintf(fout, "\tL1C_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1C_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }

    // L1T
    if(!m_shader_config->m_L1T_config.disabled()){
        total_css.clear();
        css.clear();
        fprintf(fout, "L1T_cache:\n");
        for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
            m_cluster[i]->get_L1T_sub_stats(css);
            total_css += css;
        }
        fprintf(fout, "\tL1T_total_cache_accesses = %u\n", total_css.accesses);
        fprintf(fout, "\tL1T_total_cache_misses = %u\n", total_css.misses);
        fprintf(fout, "\tL1T_total_cache_cold_misses = %u\n", total_css.cold_misses);
        if(total_css.accesses > 0){
            fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n", (double)total_css.misses / (double)total_css.accesses);
            fprintf(fout, "\tL1T_total_cache_cold_miss_rate = %.4lf\n", (double)total_css.cold_misses / (double)total_css.misses);
        }
        fprintf(fout, "\tL1T_total_cache_pending_hits = %u\n", total_css.pending_hits);
        fprintf(fout, "\tL1T_total_cache_reservation_fails = %u\n", total_css.res_fails);
    }
}

void gpgpu_sim::shader_print_l1_miss_stat( FILE *fout ) const
{
   unsigned total_d1_misses = 0, total_d1_accesses = 0;
   for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
         unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
         m_cluster[ i ]->print_cache_stats( fout, cluster_d1_accesses, custer_d1_misses );
         total_d1_misses += custer_d1_misses;
         total_d1_accesses += cluster_d1_accesses;
   }
   fprintf( fout, "total_dl1_misses=%d\n", total_d1_misses );
   fprintf( fout, "total_dl1_accesses=%d\n", total_d1_accesses );
   fprintf( fout, "total_dl1_miss_rate= %f\n", (float)total_d1_misses / (float)total_d1_accesses );
   /*
   fprintf(fout, "THD_INSN_AC: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_access_ac(i));
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i) );
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_access_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   */
}

void warp_inst_t::print( FILE *fout ) const
{
    if (empty() ) {
        fprintf(fout,"bubble\n" );
        return;
    } else 
        fprintf(fout,"0x%04x ", pc );
    fprintf(fout, "w%02d[", m_warp_id);
    for (unsigned j=0; j<m_config->warp_size; j++)
        fprintf(fout, "%c", (active(j)?'1':'0') );
    fprintf(fout, "]: ");
    ptx_print_insn( pc, fout );
    fprintf(fout, "\n");
}
void shader_core_ctx::incexecstat(warp_inst_t *&inst)
{
    if(inst->mem_op==TEX)
        inctex_stat(inst->active_count(),1);

    // Latency numbers for next operations are used to scale the power values
    // for special operations, according observations from microbenchmarking
    // TODO: put these numbers in the xml configuration

    switch(inst->sp_op){
    case INT__OP:
        incialu_stat(inst->active_count(),25);
        break;
    case INT_MUL_OP:
        incimul_stat(inst->active_count(),7.2);
        break;
    case INT_MUL24_OP:
        incimul24_stat(inst->active_count(),4.2);
        break;
    case INT_MUL32_OP:
        incimul32_stat(inst->active_count(),4);
        break;
    case INT_DIV_OP:
        incidiv_stat(inst->active_count(),40);
        break;
    case FP__OP:
        incfpalu_stat(inst->active_count(),1);
        break;
    case FP_MUL_OP:
        incfpmul_stat(inst->active_count(),1.8);
        break;
    case FP_DIV_OP:
        incfpdiv_stat(inst->active_count(),48);
        break;
    case FP_SQRT_OP:
        inctrans_stat(inst->active_count(),25);
        break;
    case FP_LG_OP:
        inctrans_stat(inst->active_count(),35);
        break;
    case FP_SIN_OP:
        inctrans_stat(inst->active_count(),12);
        break;
    case FP_EXP_OP:
        inctrans_stat(inst->active_count(),35);
        break;
    default:
        break;
    }
}
void shader_core_ctx::print_stage(unsigned int stage, FILE *fout ) const
{
   m_pipeline_reg[stage].print(fout);
   //m_pipeline_reg[stage].print(fout);
}

void shader_core_ctx::display_simt_state(FILE *fout, int mask ) const
{
    if ( (mask & 4) && m_config->model == POST_DOMINATOR ) {
       fprintf(fout,"per warp SIMT control-flow state:\n");
       unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
       for (unsigned i=0; i < n; i++) {
          unsigned nactive = 0;
          for (unsigned j=0; j<m_config->warp_size; j++ ) {
             unsigned tid = i*m_config->warp_size + j;
             int done = ptx_thread_done(tid);
             nactive += (ptx_thread_done(tid)?0:1);
             if ( done && (mask & 8) ) {
                unsigned done_cycle = m_thread[tid]->donecycle();
                if ( done_cycle ) {
                   printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
                }
             }
          }
          if ( nactive == 0 ) {
             continue;
          }
          m_simt_stack[i]->print(fout);
       }
       fprintf(fout,"\n");
    }
}

void ldst_unit::print(FILE *fout) const
{
    fprintf(fout,"LD/ST unit  = ");
    m_dispatch_reg->print(fout);
    if ( m_mem_rc != NO_RC_FAIL ) {
        fprintf(fout,"              LD/ST stall condition: ");
        switch ( m_mem_rc ) {
        case BK_CONF:        fprintf(fout,"BK_CONF"); break;
        case MSHR_RC_FAIL:   fprintf(fout,"MSHR_RC_FAIL"); break;
        case ICNT_RC_FAIL:   fprintf(fout,"ICNT_RC_FAIL"); break;
        case COAL_STALL:     fprintf(fout,"COAL_STALL"); break;
        case WB_ICNT_RC_FAIL: fprintf(fout,"WB_ICNT_RC_FAIL"); break;
        case WB_CACHE_RSRV_FAIL: fprintf(fout,"WB_CACHE_RSRV_FAIL"); break;
        case N_MEM_STAGE_STALL_TYPE: fprintf(fout,"N_MEM_STAGE_STALL_TYPE"); break;
        default: abort();
        }
        fprintf(fout,"\n");
    }
    fprintf(fout,"LD/ST wb    = ");
    m_next_wb.print(fout);
    fprintf(fout, "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                  m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );
    fprintf(fout,"Pending register writes:\n");
    std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> >::const_iterator w;
    for( w=m_pending_writes.begin(); w!=m_pending_writes.end(); w++ ) {
        unsigned warp_id = w->first;
        const std::map<unsigned/*regnum*/,unsigned/*count*/> &warp_info = w->second;
        if( warp_info.empty() ) 
            continue;
        fprintf(fout,"  w%2u : ", warp_id );
        std::map<unsigned/*regnum*/,unsigned/*count*/>::const_iterator r;
        for( r=warp_info.begin(); r!=warp_info.end(); ++r ) {
            fprintf(fout,"  %u(%u)", r->first, r->second );
        }
        fprintf(fout,"\n");
    }
    m_L1C->display_state(fout);
    m_L1T->display_state(fout);
    if( !m_config->m_L1D_config.disabled() )
        m_L1D->display_state(fout);
    fprintf(fout,"LD/ST response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void shader_core_ctx::display_pipeline(FILE *fout, int print_mem, int mask ) const
{
   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, m_not_completed);
   fprintf(fout, "=================================================\n");

   dump_warp_state(fout);
   fprintf(fout,"\n");

   m_L1I->display_state(fout);

   fprintf(fout, "IF/ID       = ");
   if( !m_inst_fetch_buffer.m_valid )
       fprintf(fout,"bubble\n");
   else {
       fprintf(fout,"w%2u : pc = 0x%x, nbytes = %u\n", 
               m_inst_fetch_buffer.m_warp_id,
               m_inst_fetch_buffer.m_pc, 
               m_inst_fetch_buffer.m_nbytes );
   }
   fprintf(fout,"\nibuffer status:\n");
   for( unsigned i=0; i<m_config->max_warps_per_shader; i++) {
       if( !m_warp[i].ibuffer_empty() ) 
           m_warp[i].print_ibuffer(fout);
   }
   fprintf(fout,"\n");
   display_simt_state(fout,mask);
   fprintf(fout, "-------------------------- Scoreboard\n");
   m_scoreboard->printContents();
/*
   fprintf(fout,"ID/OC (SP)  = ");
   print_stage(ID_OC_SP, fout);
   fprintf(fout,"ID/OC (SFU) = ");
   print_stage(ID_OC_SFU, fout);
   fprintf(fout,"ID/OC (MEM) = ");
   print_stage(ID_OC_MEM, fout);
*/
   fprintf(fout, "-------------------------- OP COL\n");
   m_operand_collector.dump(fout);
/* fprintf(fout, "OC/EX (SP)  = ");
   print_stage(OC_EX_SP, fout);
   fprintf(fout, "OC/EX (SFU) = ");
   print_stage(OC_EX_SFU, fout);
   fprintf(fout, "OC/EX (MEM) = ");
   print_stage(OC_EX_MEM, fout);
*/
   fprintf(fout, "-------------------------- Pipe Regs\n");

   for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
       fprintf(fout,"--- %s ---\n",pipeline_stage_name_decode[i]);
       print_stage(i,fout);fprintf(fout,"\n");
   }

   fprintf(fout, "-------------------------- Fu\n");
   for( unsigned n=0; n < m_num_function_units; n++ ){
       m_fu[n]->print(fout);
       fprintf(fout, "---------------\n");
   }
   fprintf(fout, "-------------------------- other:\n");

   for(unsigned i=0; i<num_result_bus; i++){
       std::string bits = m_result_bus[i]->to_string();
       fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str() );
   }
   fprintf(fout, "EX/WB      = ");
   print_stage(EX_WB, fout);
   fprintf(fout, "\n");
   fprintf(fout, "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                 m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );

   if( m_active_threads.count() <= 2*m_config->warp_size ) {
       fprintf(fout,"Active Threads : ");
       unsigned last_warp_id = -1;
       for(unsigned tid=0; tid < m_active_threads.size(); tid++ ) {
           unsigned warp_id = tid/m_config->warp_size;
           if( m_active_threads.test(tid) ) {
               if( warp_id != last_warp_id ) {
                   fprintf(fout,"\n  warp %u : ", warp_id );
                   last_warp_id=warp_id;
               }
               fprintf(fout,"%u ", tid );
           }
       }
   }

}

unsigned int shader_core_config::max_cta( const kernel_info_t &k ) const
{
	if (gKernelName == 0)
	{
		gKernelName = new char[strlen(k.name().c_str()) + 1];
		strcpy(gKernelName, k.name().c_str());
	}
	if (gBmName == 0)
	{
		char bmNameStr[1024];
		strcpy(bmNameStr, get_current_dir_name());
		char* bmNamePtr = strrchr(bmNameStr, '/');
		bmNamePtr++;
		gBmName = new char[strlen(bmNamePtr) + 1];
		strcpy(gBmName, bmNamePtr);
		printf("BM = %s, Kernel = %s\n", gBmName, gKernelName);
	}

   unsigned threads_per_cta  = k.threads_per_cta();
   const class function_info *kernel = k.entry();
   unsigned int padded_cta_size = threads_per_cta;
   if (padded_cta_size%warp_size) 
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

   //Limit by n_threads/shader
   unsigned int result_thread = n_thread_per_shader / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info(kernel);

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = gpgpu_shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = gpgpu_shader_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = max_cta_per_core;

   unsigned result = result_thread;
   result = gs_min2(result, result_shmem);
   result = gs_min2(result, result_regs);
   result = gs_min2(result, result_cta);

   static const struct gpgpu_ptx_sim_kernel_info* last_kinfo = NULL;
   if (last_kinfo != kernel_info) {   //Only print out stats if kernel_info struct changes
      last_kinfo = kernel_info;
      printf ("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
      if (result == result_thread) printf (" threads");
      if (result == result_shmem) printf (" shmem");
      if (result == result_regs) printf (" regs");
      if (result == result_cta) printf (" cta_limit");
      printf ("\n");
	printf("Total num of TBs %u\n", (unsigned int)k.num_blocks());
	gTotalNumOfTBsInGrid = k.num_blocks();
   }

    //gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep all cores busy    
    if( k.num_blocks() < result*num_shader() ) { 
       result = k.num_blocks() / num_shader();
       if (k.num_blocks() % num_shader())
          result++;
    }

    assert( result <= MAX_CTA_PER_SHADER );
    if (result < 1) {
       printf ("GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader has.\n");
       abort();
    }

    return result;
}

void shader_core_ctx::cycle()
{
    m_stats->shader_cycles[m_sid]++;
    writeback();
    execute();
    read_operands();
    issue();
    decode();
    fetch();
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush()
{
   m_ldst_unit->flush();
}

// modifiers
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() 
{
   std::list<op_t> result;  // a list of registers that (a) are in different register banks, (b) do not go to the same operand collector

   int input;
   int output;
   int _inputs = m_num_banks;
   int _outputs = m_num_collectors;
   int _square = ( _inputs > _outputs ) ? _inputs : _outputs;
   assert(_square > 0);
   int _pri = (int)m_last_cu;

   // Clear matching
   for ( int i = 0; i < _inputs; ++i ) 
      _inmatch[i] = -1;
   for ( int j = 0; j < _outputs; ++j ) 
      _outmatch[j] = -1;

   for( unsigned i=0; i<m_num_banks; i++) {
      for( unsigned j=0; j<m_num_collectors; j++) {
         assert( i < (unsigned)_inputs );
         assert( j < (unsigned)_outputs );
         _request[i][j] = 0;
      }
      if( !m_queue[i].empty() ) {
         const op_t &op = m_queue[i].front();
         int oc_id = op.get_oc_id();
         assert( i < (unsigned)_inputs );
         assert( oc_id < _outputs );
         _request[i][oc_id] = 1;
      }
      if( m_allocated_bank[i].is_write() ) {
         assert( i < (unsigned)_inputs );
         _inmatch[i] = 0; // write gets priority
      }
   }

   ///// wavefront allocator from booksim... --->
   
   // Loop through diagonals of request matrix

   for ( int p = 0; p < _square; ++p ) {
      output = ( _pri + p ) % _square;

      // Step through the current diagonal
      for ( input = 0; input < _inputs; ++input ) {
          assert( input < _inputs );
          assert( output < _outputs );
         if ( ( output < _outputs ) && 
              ( _inmatch[input] == -1 ) && 
              ( _outmatch[output] == -1 ) &&
              ( _request[input][output]/*.label != -1*/ ) ) {
            // Grant!
            _inmatch[input] = output;
            _outmatch[output] = input;
         }

         output = ( output + 1 ) % _square;
      }
   }

   // Round-robin the priority diagonal
   _pri = ( _pri + 1 ) % _square;

   /// <--- end code from booksim

   m_last_cu = _pri;
   for( unsigned i=0; i < m_num_banks; i++ ) {
      if( _inmatch[i] != -1 ) {
         if( !m_allocated_bank[i].is_write() ) {
            unsigned bank = (unsigned)i;
            op_t &op = m_queue[bank].front();
            result.push_back(op);
            m_queue[bank].pop_front();
         }
      }
   }

   return result;
}

barrier_set_t::barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core )
{
   m_max_warps_per_core = max_warps_per_core;
   m_max_cta_per_core = max_cta_per_core;
   if( max_warps_per_core > WARP_PER_CTA_MAX ) {
      printf("ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or warps per cta in gpgpusim.config\n",
             WARP_PER_CTA_MAX, max_warps_per_core );
      exit(1);
   }
   m_warp_active.reset();
   m_warp_at_barrier.reset();
}

// during cta allocation
void barrier_set_t::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   assert( cta_id < m_max_cta_per_core );
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   assert( w == m_cta_to_warps.end() ); // cta should not already be active or allocated barrier resources
   m_cta_to_warps[cta_id] = warps;
   assert( m_cta_to_warps.size() <= m_max_cta_per_core ); // catch cta's that were not properly deallocated
  
   m_warp_active |= warps;
   m_warp_at_barrier &= ~warps;
}

// during cta deallocation
void barrier_set_t::deallocate_barrier( unsigned cta_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   if( w == m_cta_to_warps.end() )
      return;
   warp_set_t warps = w->second;
   warp_set_t at_barrier = warps & m_warp_at_barrier;
   assert( at_barrier.any() == false ); // no warps stuck at barrier
   warp_set_t active = warps & m_warp_active;
   assert( active.any() == false ); // no warps in CTA still running
   m_warp_active &= ~warps;
   m_warp_at_barrier &= ~warps;
   m_cta_to_warps.erase(w);
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier( unsigned cta_id, unsigned warp_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);

   if( w == m_cta_to_warps.end() ) { // cta is active
      printf("ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n", cta_id, gpu_tot_sim_cycle, gpu_sim_cycle );
      dump();
      abort();
   }
   assert( w->second.test(warp_id) == true ); // warp is in cta

   m_warp_at_barrier.set(warp_id);

   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

// fetching a warp
bool barrier_set_t::available_for_fetch( unsigned warp_id ) const
{
   return m_warp_active.test(warp_id) && m_warp_at_barrier.test(warp_id);
}

// warp reaches exit 
void barrier_set_t::warp_exit( unsigned warp_id )
{
   // caller needs to verify all threads in warp are done, e.g., by checking PDOM stack to 
   // see it has only one entry during exit_impl()
   m_warp_active.reset(warp_id);

   // test for barrier release 
   cta_to_warp_t::iterator w=m_cta_to_warps.begin(); 
   for (; w != m_cta_to_warps.end(); ++w) {
      if (w->second.test(warp_id) == true) break; 
   }
   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier( unsigned warp_id ) const
{ 
   return m_warp_at_barrier.test(warp_id);
}

void barrier_set_t::dump() const
{
   printf( "barrier set information\n");
   printf( "  m_max_cta_per_core = %u\n",  m_max_cta_per_core );
   printf( "  m_max_warps_per_core = %u\n", m_max_warps_per_core );
   printf( "  cta_to_warps:\n");
   
   cta_to_warp_t::const_iterator i;
   for( i=m_cta_to_warps.begin(); i!=m_cta_to_warps.end(); i++ ) {
      unsigned cta_id = i->first;
      warp_set_t warps = i->second;
      printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str() );
   }
   printf("  warp_active: %s\n", m_warp_active.to_string().c_str() );
   printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str() );
   fflush(stdout); 
}

void shader_core_ctx::warp_exit( unsigned warp_id )
{
    bool done = true;
    for (    unsigned i = warp_id*get_config()->warp_size;
            i < (warp_id+1)*get_config()->warp_size;
            i++ ) {

//        if(this->m_thread[i]->m_functional_model_thread_state && this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
//            done = false;
//        }


        if (m_thread[i] && !m_thread[i]->is_done()) done = false;
    }
    //if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
    //if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
    if (done)
        m_barriers.warp_exit( warp_id );
}

bool shader_core_ctx::warp_waiting_at_barrier( unsigned warp_id ) const
{
   return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool shader_core_ctx::warp_waiting_at_mem_barrier( unsigned warp_id ) 
{
   if( !m_warp[warp_id].get_membar() ) 
      return false;
   if( !m_scoreboard->pendingWrites(warp_id) ) {
      m_warp[warp_id].clear_membar();
      return false;
   }
   return true;
}

void shader_core_ctx::set_max_cta( const kernel_info_t &kernel ) 
{
    // calculate the max cta count and cta size for local memory address mapping
    kernel_max_cta_per_shader = m_config->max_cta(kernel);
    unsigned int gpu_cta_size = kernel.threads_per_cta();
    kernel_padded_threads_per_cta = (gpu_cta_size%m_config->warp_size) ? 
        m_config->warp_size*((gpu_cta_size/m_config->warp_size)+1) : 
        gpu_cta_size;
}

void shader_core_ctx::decrement_atomic_count( unsigned wid, unsigned n )
{
   assert( m_warp[wid].get_n_atomic() >= n );
   m_warp[wid].dec_n_atomic(n);
}


bool shader_core_ctx::fetch_unit_response_buffer_full() const
{
    return false;
}

void shader_core_ctx::accept_fetch_response( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_L1I->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
}

bool shader_core_ctx::ldst_unit_response_buffer_full() const
{
    return m_ldst_unit->response_buffer_full();
}

void shader_core_ctx::accept_ldst_unit_response(mem_fetch * mf) 
{
   m_ldst_unit->fill(mf);
}

void shader_core_ctx::store_ack( class mem_fetch *mf )
{
    assert( mf->get_type() == WRITE_ACK  || ( m_config->gpgpu_perfect_mem && mf->get_is_write() ) );
    unsigned warp_id = mf->get_wid();
    m_warp[warp_id].dec_store_req();
	gStoreReqInProgress--;
	// printf("%llu: sm %u done store for warp %u, num store reqs in progress = %u\n", gpu_sim_cycle, this->get_sid(), warp_id, gStoreReqInProgress);
}

void shader_core_ctx::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   m_ldst_unit->print_cache_stats( fp, dl1_accesses, dl1_misses );
}

void shader_core_ctx::get_cache_stats(cache_stats &cs){
    // Adds stats from each cache to 'cs'
    cs += m_L1I->get_stats(); // Get L1I stats
    m_ldst_unit->get_cache_stats(cs); // Get L1D, L1C, L1T stats
}

void shader_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const{
    if(m_L1I)
        m_L1I->get_sub_stats(css);
}
void shader_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1D_sub_stats(css);
}
void shader_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1C_sub_stats(css);
}
void shader_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    m_ldst_unit->get_L1T_sub_stats(css);
}

void shader_core_ctx::get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const{
    n_simt_to_mem += m_stats->n_simt_to_mem[m_sid];
    n_mem_to_simt += m_stats->n_mem_to_simt[m_sid];
}

bool shd_warp_t::functional_done() const
{
    return get_n_completed() == m_warp_size;
}

bool shd_warp_t::hardware_done() const
{
    return functional_done() && stores_done() && !inst_in_pipeline(); 
}

bool shd_warp_t::waiting() 
{
    if ( functional_done() ) {
        // waiting to be initialized with a kernel
        return true;
    } else if ( m_shader->warp_waiting_at_barrier(m_warp_id) ) {
        // waiting for other warps in CTA to reach barrier
        return true;
    } else if ( m_shader->warp_waiting_at_mem_barrier(m_warp_id) ) {
        // waiting for memory barrier
        return true;
    } else if ( m_n_atomic >0 ) {
        // waiting for atomic operation to complete at memory:
        // this stall is not required for accurate timing model, but rather we
        // stall here since if a call/return instruction occurs in the meantime
        // the functional execution of the atomic when it hits DRAM can cause
        // the wrong register to be read.
        return true;
    }
    return false;
}

void shd_warp_t::print( FILE *fout ) const
{
    if( !done_exit() ) {
        fprintf( fout, "w%02u npc: 0x%04x, done:%c%c%c%c:%2u i:%u s:%u a:%u (done: ", 
                m_warp_id,
                m_next_pc,
                (functional_done()?'f':' '),
                (stores_done()?'s':' '),
                (inst_in_pipeline()?' ':'i'),
                (done_exit()?'e':' '),
                n_completed,
                m_inst_in_pipeline, 
                m_stores_outstanding,
                m_n_atomic );
        for (unsigned i = m_warp_id*m_warp_size; i < (m_warp_id+1)*m_warp_size; i++ ) {
          if ( m_shader->ptx_thread_done(i) ) fprintf(fout,"1");
          else fprintf(fout,"0");
          if ( (((i+1)%4) == 0) && (i+1) < (m_warp_id+1)*m_warp_size ) 
             fprintf(fout,",");
        }
        fprintf(fout,") ");
        fprintf(fout," active=%s", m_active_threads.to_string().c_str() );
        fprintf(fout," last fetched @ %5llu", m_last_fetch);
        if( m_imiss_pending ) 
            fprintf(fout," i-miss pending");
        fprintf(fout,"\n");
    }
}

void shd_warp_t::print_ibuffer( FILE *fout ) const
{
    fprintf(fout,"  ibuffer[%2u] : ", m_warp_id );
    for( unsigned i=0; i < IBUFFER_SIZE; i++) {
        const inst_t *inst = m_ibuffer[i].m_inst;
        if( inst ) inst->print_insn(fout);
        else if( m_ibuffer[i].m_valid ) 
           fprintf(fout," <invalid instruction> ");
        else fprintf(fout," <empty> ");
    }
    fprintf(fout,"\n");
}

void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu, unsigned num_dispatch){
    m_cus[set_id].reserve(num_cu); //this is necessary to stop pointers in m_cu from being invalid do to a resize;
    for (unsigned i = 0; i < num_cu; i++) {
        m_cus[set_id].push_back(collector_unit_t());
        m_cu.push_back(&m_cus[set_id].back());
    }
    // for now each collector set gets dedicated dispatch units.
    for (unsigned i = 0; i < num_dispatch; i++) {
        m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    }
}


void opndcoll_rfu_t::add_port(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
{
    //m_num_ports++;
    //m_num_collectors += num_collector_units;
    //m_input.resize(m_num_ports);
    //m_output.resize(m_num_ports);
    //m_num_collector_units.resize(m_num_ports);
    //m_input[m_num_ports-1]=input_port;
    //m_output[m_num_ports-1]=output_port;
    //m_num_collector_units[m_num_ports-1]=num_collector_units;
    m_in_ports.push_back(input_port_t(input,output,cu_sets));
}

void opndcoll_rfu_t::init( unsigned num_banks, shader_core_ctx *shader )
{
   m_shader=shader;
   m_arbiter.init(m_cu.size(),num_banks);
   //for( unsigned n=0; n<m_num_ports;n++ ) 
   //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
   m_num_banks = num_banks;
   m_bank_warp_shift = 0; 
   m_warp_size = shader->get_config()->warp_size;
   m_bank_warp_shift = (unsigned)(int) (log(m_warp_size+0.5) / log(2.0));
   assert( (m_bank_warp_shift == 5) || (m_warp_size != 32) );

   for( unsigned j=0; j<m_cu.size(); j++) {
       m_cu[j]->init(j,num_banks,m_bank_warp_shift,shader->get_config(),this);
   }
   m_initialized=true;
}

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += wid;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const warp_inst_t &inst )
{
   assert( !inst.empty() );
   std::list<unsigned> regs = m_shader->get_regs_written(inst);
   std::list<unsigned>::iterator r;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,inst.warp_id(),m_num_banks,m_bank_warp_shift);
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&inst,reg,m_num_banks,m_bank_warp_shift));
      } else {
          return false;
      }
   }
   for(unsigned i=0;i<(unsigned)regs.size();i++){
          if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
              unsigned active_count=0;
              for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
                  for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
                      if(inst.get_active_mask().test(i+j)){
                          active_count+=m_shader->get_config()->n_regfile_gating_group;
                          break;
                      }
                  }
              }
              m_shader->incregfile_writes(active_count);
          }else{
              m_shader->incregfile_writes(m_shader->get_config()->warp_size);//inst.active_count());
          }
   }
   return true;
}

void opndcoll_rfu_t::dispatch_ready_cu()
{
   for( unsigned p=0; p < m_dispatch_units.size(); ++p ) {
      dispatch_unit_t &du = m_dispatch_units[p];
      collector_unit_t *cu = du.find_ready();
      if( cu ) {
         for(unsigned i=0;i<(cu->get_num_operands()-cu->get_num_regs());i++){
             if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
                 unsigned active_count=0;
                 for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
                     for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
                         if(cu->get_active_mask().test(i+j)){
                             active_count+=m_shader->get_config()->n_regfile_gating_group;
                             break;
                         }
                     }
                 }
                 m_shader->incnon_rf_operands(active_count);
             }else{
             m_shader->incnon_rf_operands(m_shader->get_config()->warp_size);//cu->get_active_count());
             }
        }
         cu->dispatch();
      }
   }
}

void opndcoll_rfu_t::allocate_cu( unsigned port_num )
{
   input_port_t& inp = m_in_ports[port_num];
   for (unsigned i = 0; i < inp.m_in.size(); i++) {
       if( (*inp.m_in[i]).has_ready() ) {
          //find a free cu 
          for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
              std::vector<collector_unit_t> & cu_set = m_cus[inp.m_cu_sets[j]];
          bool allocated = false;
              for (unsigned k = 0; k < cu_set.size(); k++) {
                  if(cu_set[k].is_free()) {
                     collector_unit_t *cu = &cu_set[k];
                     allocated = cu->allocate(inp.m_in[i],inp.m_out[i]);
                     m_arbiter.add_read_requests(cu);
                     break;
                  }
              }
              if (allocated) break; //cu has been allocated, no need to search more.
          }
          break; // can only service a single input, if it failed it will fail for others.
       }
   }
}

void opndcoll_rfu_t::allocate_reads()
{
   // process read requests that do not have conflicts
   std::list<op_t> allocated = m_arbiter.allocate_reads();
   std::map<unsigned,op_t> read_ops;
   for( std::list<op_t>::iterator r=allocated.begin(); r!=allocated.end(); r++ ) {
      const op_t &rr = *r;
      unsigned reg = rr.get_reg();
      unsigned wid = rr.get_wid();
      unsigned bank = register_bank(reg,wid,m_num_banks,m_bank_warp_shift);
      m_arbiter.allocate_for_read(bank,rr);
      read_ops[bank] = rr;
   }
   std::map<unsigned,op_t>::iterator r;
   for(r=read_ops.begin();r!=read_ops.end();++r ) {
      op_t &op = r->second;
      unsigned cu = op.get_oc_id();
      unsigned operand = op.get_operand();
      m_cu[cu]->collect_operand(operand);
      if(m_shader->get_config()->gpgpu_clock_gated_reg_file){
          unsigned active_count=0;
          for(unsigned i=0;i<m_shader->get_config()->warp_size;i=i+m_shader->get_config()->n_regfile_gating_group){
              for(unsigned j=0;j<m_shader->get_config()->n_regfile_gating_group;j++){
                  if(op.get_active_mask().test(i+j)){
                      active_count+=m_shader->get_config()->n_regfile_gating_group;
                      break;
                  }
              }
          }
          m_shader->incregfile_reads(active_count);
      }else{
          m_shader->incregfile_reads(m_shader->get_config()->warp_size);//op.get_active_count());
      }
  }
} 

bool opndcoll_rfu_t::collector_unit_t::ready() const 
{ 
   return (!m_free) && m_not_ready.none() && (*m_output_register).has_free(); 
}

void opndcoll_rfu_t::collector_unit_t::dump(FILE *fp, const shader_core_ctx *shader ) const
{
   if( m_free ) {
      fprintf(fp,"    <free>\n");
   } else {
      m_warp->print(fp);
      for( unsigned i=0; i < MAX_REG_OPERANDS*2; i++ ) {
         if( m_not_ready.test(i) ) {
            std::string r = m_src_op[i].get_reg_string();
            fprintf(fp,"    '%s' not ready\n", r.c_str() );
         }
      }
   }
}

void opndcoll_rfu_t::collector_unit_t::init( unsigned n, 
                                             unsigned num_banks, 
                                             unsigned log2_warp_size,
                                             const core_config *config,
                                             opndcoll_rfu_t *rfu ) 
{ 
   m_rfu=rfu;
   m_cuid=n; 
   m_num_banks=num_banks;
   assert(m_warp==NULL); 
   m_warp = new warp_inst_t(config);
   m_bank_warp_shift=log2_warp_size;
}

bool opndcoll_rfu_t::collector_unit_t::allocate( register_set* pipeline_reg_set, register_set* output_reg_set ) 
{
   assert(m_free);
   assert(m_not_ready.none());
   m_free = false;
   m_output_register = output_reg_set;
   warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
   if( (pipeline_reg) and !((*pipeline_reg)->empty()) ) {
      m_warp_id = (*pipeline_reg)->warp_id();
      for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
         int reg_num = (*pipeline_reg)->arch_reg.src[op]; // this math needs to match that used in function_info::ptx_decode_inst
         if( reg_num >= 0 ) { // valid register
            m_src_op[op] = op_t( this, op, reg_num, m_num_banks, m_bank_warp_shift );
            m_not_ready.set(op);
         } else 
            m_src_op[op] = op_t();
      }
      //move_warp(m_warp,*pipeline_reg);
      pipeline_reg_set->move_out_to(m_warp);
      return true;
   }
   return false;
}

void opndcoll_rfu_t::collector_unit_t::dispatch()
{
   assert( m_not_ready.none() );
   //move_warp(*m_output_register,m_warp);
   m_output_register->move_in(m_warp);
   m_free=true;
   m_output_register = NULL;
   for( unsigned i=0; i<MAX_REG_OPERANDS*2;i++)
      m_src_op[i].reset();
}

simt_core_cluster::simt_core_cluster( class gpgpu_sim *gpu, 
                                      unsigned cluster_id, 
                                      const struct shader_core_config *config, 
                                      const struct memory_config *mem_config,
                                      shader_core_stats *stats, 
                                      class memory_stats_t *mstats )
{
    m_config = config;
    m_cta_issue_next_core=m_config->n_simt_cores_per_cluster-1; // this causes first launch to use hw cta 0
    m_cluster_id=cluster_id;
    m_gpu = gpu;
    m_stats = stats;
    m_memory_stats = mstats;
    m_core = new shader_core_ctx*[ config->n_simt_cores_per_cluster ];
    for( unsigned i=0; i < config->n_simt_cores_per_cluster; i++ ) {
        unsigned sid = m_config->cid_to_sid(i,m_cluster_id);
        m_core[i] = new shader_core_ctx(gpu,this,sid,m_cluster_id,config,mem_config,stats);
        m_core_sim_order.push_back(i); 
    }
}

void simt_core_cluster::core_cycle()
{
    for( std::list<unsigned>::iterator it = m_core_sim_order.begin(); it != m_core_sim_order.end(); ++it ) {
        m_core[*it]->cycle();
    }

    if (m_config->simt_core_sim_order == 1) {
        m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order, m_core_sim_order.begin()); 
    }
}

void simt_core_cluster::reinit()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->reinit(0,m_config->n_thread_per_shader,true);
}

unsigned simt_core_cluster::max_cta( const kernel_info_t &kernel )
{
    return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel);
}

unsigned simt_core_cluster::get_not_completed() const
{
    unsigned not_completed=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        not_completed += m_core[i]->get_not_completed();
    return not_completed;
}

void simt_core_cluster::print_not_completed( FILE *fp ) const
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned not_completed=m_core[i]->get_not_completed();
        unsigned sid=m_config->cid_to_sid(i,m_cluster_id);
        fprintf(fp,"%u(%u) ", sid, not_completed );
    }
}

unsigned simt_core_cluster::get_n_active_cta() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        n += m_core[i]->get_n_active_cta();
    return n;
}

unsigned simt_core_cluster::get_n_active_sms() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ )
        n += m_core[i]->isactive();
    return n;
}

unsigned simt_core_cluster::issue_block2core()
{
    unsigned num_blocks_issued=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned core = (i+m_cta_issue_next_core+1)%m_config->n_simt_cores_per_cluster;
        if( m_core[core]->get_not_completed() == 0 ) {
            if( m_core[core]->get_kernel() == NULL ) {
                kernel_info_t *k = m_gpu->select_kernel();
                if( k ) 
                    m_core[core]->set_kernel(k);
            }
        }
        kernel_info_t *kernel = m_core[core]->get_kernel();
        if( kernel && !kernel->no_more_ctas_to_run() && (m_core[core]->get_n_active_cta() < m_config->max_cta(*kernel)) ) {
            m_core[core]->issue_block2core(*kernel);
            num_blocks_issued++;
            m_cta_issue_next_core=core; 
            break;
        }
    }
    return num_blocks_issued;
}

void simt_core_cluster::cache_flush()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->cache_flush();
}

bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write)
{
    unsigned request_size = size;
    if (!write) 
        request_size = READ_PACKET_SIZE;
    return ! ::icnt_has_buffer(m_cluster_id, request_size);
}

void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf)
{
    // stats
    if (mf->get_is_write()) m_stats->made_write_mfs++;
    else m_stats->made_read_mfs++;
    switch (mf->get_access_type()) {
    case CONST_ACC_R: m_stats->gpgpu_n_mem_const++; break;
    case TEXTURE_ACC_R: m_stats->gpgpu_n_mem_texture++; break;
    case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++; break;
    case GLOBAL_ACC_W: m_stats->gpgpu_n_mem_write_global++; break;
    case LOCAL_ACC_R: m_stats->gpgpu_n_mem_read_local++; break;
    case LOCAL_ACC_W: m_stats->gpgpu_n_mem_write_local++; break;
    case INST_ACC_R: m_stats->gpgpu_n_mem_read_inst++; break;
    case L1_WRBK_ACC: m_stats->gpgpu_n_mem_write_global++; break;
    case L2_WRBK_ACC: m_stats->gpgpu_n_mem_l2_writeback++; break;
    case L1_WR_ALLOC_R: m_stats->gpgpu_n_mem_l1_write_allocate++; break;
    case L2_WR_ALLOC_R: m_stats->gpgpu_n_mem_l2_write_allocate++; break;
    default: assert(0);
    }

   // The packet size varies depending on the type of request: 
   // - For write request and atomic request, the packet contains the data 
   // - For read request (i.e. not write nor atomic), the packet only has control metadata
   unsigned int packet_size = mf->size(); 
   if (!mf->get_is_write() && !mf->isatomic()) {
      packet_size = mf->get_ctrl_size(); 
   }
   m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size); 
   unsigned destination = mf->get_sub_partition_id();
   mf->set_status(IN_ICNT_TO_MEM,gpu_sim_cycle+gpu_tot_sim_cycle);
   if (!mf->get_is_write() && !mf->isatomic())
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->get_ctrl_size() );
   else 
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->size());
}

void simt_core_cluster::icnt_cycle()
{
    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        unsigned cid = m_config->sid_to_cid(mf->get_sid());
        if( mf->get_access_type() == INST_ACC_R ) {
            // instruction fetch response
            if( !m_core[cid]->fetch_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_core[cid]->accept_fetch_response(mf);
            }
        } else {
            // data response
            if( !m_core[cid]->ldst_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_memory_stats->memlatstat_read_done(mf);
                m_core[cid]->accept_ldst_unit_response(mf);
            }
        }
    }
    if( m_response_fifo.size() < m_config->n_simt_ejection_buffer_size ) {
        mem_fetch *mf = (mem_fetch*) ::icnt_pop(m_cluster_id);
        if (!mf) 
            return;
        assert(mf->get_tpc() == m_cluster_id);
        assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK );

        // The packet size varies depending on the type of request: 
        // - For read request and atomic request, the packet contains the data 
        // - For write-ack, the packet only has control metadata
        unsigned int packet_size = (mf->get_is_write())? mf->get_ctrl_size() : mf->size(); 
        m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size); 
        mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        //m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
        m_response_fifo.push_back(mf);
        m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
    }
}

void simt_core_cluster::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned cid = m_config->sid_to_cid(sid);
    m_core[cid]->get_pdom_stack_top_info(tid,pc,rpc);
}

void simt_core_cluster::display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask )
{
    m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout,print_mem,mask);

    fprintf(fout,"\n");
    fprintf(fout,"Cluster %u pipeline state\n", m_cluster_id );
    fprintf(fout,"Response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void simt_core_cluster::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const {
   for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
      m_core[ i ]->print_cache_stats( fp, dl1_accesses, dl1_misses );
   }
}

void simt_core_cluster::get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const {
    long simt_to_mem=0;
    long mem_to_simt=0;
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt);
    }
    n_simt_to_mem = simt_to_mem;
    n_mem_to_simt = mem_to_simt;
}

void simt_core_cluster::get_cache_stats(cache_stats &cs) const{
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_cache_stats(cs);
    }
}

void simt_core_cluster::get_L1I_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1I_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1D_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1C_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1C_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}
void simt_core_cluster::get_L1T_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats temp_css;
    struct cache_sub_stats total_css;
    temp_css.clear();
    total_css.clear();
    for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
        m_core[i]->get_L1T_sub_stats(temp_css);
        total_css += temp_css;
    }
    css = total_css;
}

void shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)
{
    if( inst.has_callback(t) ) 
           m_warp[inst.warp_id()].inc_n_atomic();
        if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
            new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
            unsigned num_addrs;
            num_addrs = translate_local_memaddr(inst.get_addr(t), tid, m_config->n_simt_clusters*m_config->n_simt_cores_per_cluster,
                   inst.data_size, (new_addr_type*) localaddrs );
            inst.set_addr(t, (new_addr_type*) localaddrs, num_addrs);
        }
        if ( ptx_thread_done(tid) ) {
            m_warp[inst.warp_id()].set_completed(t);
            m_warp[inst.warp_id()].ibuffer_flush();
			if (t == 0)
			{
				uint smId = this->get_sid();
				uint warpId = inst.warp_id();
				// printf("%llu: warp %u(%u), sm %u finishing\n", gpu_sim_cycle, warpId, m_warp[warpId].get_dynamic_warp_id(), smId);
				unsigned int warpIdx = smId * MAX_NUM_WARP_PER_SM + warpId;
				if (gWarpDrainTimeArray[warpIdx] == 0)
					gWarpDrainTimeArray[warpIdx] = gpu_sim_cycle;

				if ((this->get_sid() == 2) && (inst.warp_id() == 39))
					gWarpFinishing = true;
			}
        }

    // PC-Histogram Update 
    unsigned warp_id = inst.warp_id(); 
    unsigned pc = inst.pc; 
    for (unsigned t = 0; t < m_config->warp_size; t++) {
        if (inst.active(t)) {
            int tid = warp_id * m_config->warp_size + t; 
            cflog_update_thread_pc(m_sid, tid, pc);  
        }
    }
}

