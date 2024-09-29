# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import math
import os
from enum import Enum
from typing import Dict, List, Optional

import torch

from ..utils import log_on_each_pipeline_stage
from .distributed_data_parallel_config import DistributedDataParallelConfig

# HANS: Additionals
import numpy as np
import time
# from ..optimizer.clip_grads import get_grad_norm_fp32

logger = logging.getLogger(__name__)


class BufferType(Enum):
    PARAM = 1
    GRAD = 2


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Args:
        ddp_config: DistributedDataParallel config object.
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in larger ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in larger ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        gradient_scaling_factor: float,
        bucket_id: int,
    ):
        self.ddp_config = ddp_config

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id

        # HANS: Additionals
        self.iteration = 0

        # For BERT-Large
        # self.bin_edges = torch.tensor([0, 1024, 1049600, 1049602, 1051650, 1080706, 1081730, 1082754, 1083778, 2132354, 2133378, 2134402, 2135426, 6329730, 6333826, 10528130, 10529154, 10530178, 10533250, 13678978, 13680002, 14728578, 14729602, 14730626, 14731650, 18925954, 18930050, 23124354, 23125378, 23126402, 23129474, 26275202, 26276226, 27324802, 27325826, 27326850, 27327874, 31522178, 31526274, 35720578, 35721602, 35722626, 35725698, 38871426, 38872450, 39921026, 39922050, 39923074, 39924098, 44118402, 44122498, 48316802, 48317826, 48318850, 48321922, 51467650, 51468674, 52517250, 52518274, 52519298, 52520322, 56714626, 56718722, 60913026, 60914050, 60915074, 60918146, 64063874, 64064898, 65113474, 65114498, 65115522, 65116546, 69310850, 69314946, 73509250, 73510274, 73511298, 73514370, 76660098, 76661122, 77709698, 77710722, 77711746, 77712770, 81907074, 81911170, 86105474, 86106498, 86107522, 86110594, 89256322, 89257346, 90305922, 90306946, 90307970, 90308994, 94503298, 94507394, 98701698, 98702722, 98703746, 98706818, 101852546, 101853570, 102902146, 102903170, 102904194, 102905218, 107099522, 107103618, 111297922, 111298946, 111299970, 111303042, 114448770, 114449794, 115498370, 115499394, 115500418, 115501442, 119695746, 119699842, 123894146, 123895170, 123896194, 123899266, 127044994, 127046018, 128094594, 128095618, 128096642, 128097666, 132291970, 132296066, 136490370, 136491394, 136492418, 136495490, 139641218, 139642242, 140690818, 140691842, 140692866, 140693890, 144888194, 144892290, 149086594, 149087618, 149088642, 149091714, 152237442, 152238466, 153287042, 153288066, 153289090, 153290114, 157484418, 157488514, 161682818, 161683842, 161684866, 161687938, 164833666, 164834690, 165883266, 165884290, 165885314, 165886338, 170080642, 170084738, 174279042, 174280066, 174281090, 174284162, 177429890, 177430914, 178479490, 178480514, 178481538, 178482562, 182676866, 182680962, 186875266, 186876290, 186877314, 186880386, 190026114, 190027138, 191075714, 191076738, 191077762, 191078786, 195273090, 195277186, 199471490, 199472514, 199473538, 199476610, 202622338, 202623362, 203671938, 203672962, 203673986, 203675010, 207869314, 207873410, 212067714, 212068738, 212069762, 212072834, 215218562, 215219586, 216268162, 216269186, 216270210, 216271234, 220465538, 220469634, 224663938, 224664962, 224665986, 224669058, 227814786, 227815810, 228864386, 228865410, 228866434, 228867458, 233061762, 233065858, 237260162, 237261186, 237262210, 237265282, 240411010, 240412034, 241460610, 241461634, 241462658, 241463682, 245657986, 245662082, 249856386, 249857410, 249858434, 249861506, 253007234, 253008258, 254056834, 254057858, 254058882, 254059906, 258254210, 258258306, 262452610, 262453634, 262454658, 262457730, 265603458, 265604482, 266653058, 266654082, 266655106, 266656130, 270850434, 270854530, 275048834, 275049858, 275050882, 275053954, 278199682, 278200706, 279249282, 279250306, 279251330, 279252354, 283446658, 283450754, 287645058, 287646082, 287647106, 287650178, 290795906, 290796930, 291845506, 291846530, 291847554, 291848578, 296042882, 296046978, 300241282, 300242306, 300243330, 300246402, 303392130, 303393154, 304441730, 304442754, 304443778, 304445826, 304970114]).cuda()

        # For 1.7B GPT-3
        # self.bin_edges = torch.tensor([0, 2304, 4608, 6912, 21240576, 21249792, 42483456, 42485760, 42488064, 42494976, 58420224, 58422528, 63730944, 63733248, 63735552, 63737856, 84971520, 84980736, 106214400, 106216704, 106219008, 106225920, 122151168, 122153472, 127461888, 127464192, 127466496, 127468800, 148702464, 148711680, 169945344, 169947648, 169949952, 169956864, 185882112, 185884416, 191192832, 191195136, 191197440, 191199744, 212433408, 212442624, 233676288, 233678592, 233680896, 233687808, 249613056, 249615360, 254923776, 254926080, 254928384, 254930688, 276164352, 276173568, 297407232, 297409536, 297411840, 297418752, 313344000, 313346304, 318654720, 318657024, 318659328, 318661632, 339895296, 339904512, 361138176, 361140480, 361142784, 361149696, 377074944, 377077248, 382385664, 382387968, 382390272, 382392576, 403626240, 403635456, 424869120, 424871424, 424873728, 424880640, 440805888, 440808192, 446116608, 446118912, 446121216, 446123520, 467357184, 467366400, 488600064, 488602368, 488604672, 488611584, 504536832, 504539136, 509847552, 509849856, 509852160, 509854464, 531088128, 531097344, 552331008, 552333312, 552335616, 552342528, 568267776, 568270080, 573578496, 573580800, 573583104, 573585408, 594819072, 594828288, 616061952, 616064256, 616066560, 616073472, 631998720, 632001024, 637309440, 637311744, 637314048, 637316352, 658550016, 658559232, 679792896, 679795200, 679797504, 679804416, 695729664, 695731968, 701040384, 701042688, 701044992, 701047296, 722280960, 722290176, 743523840, 743526144, 743528448, 743535360, 759460608, 759462912, 764771328, 764773632, 764775936, 764778240, 786011904, 786021120, 807254784, 807257088, 807259392, 807266304, 823191552, 823193856, 828502272, 828504576, 828506880, 828509184, 849742848, 849752064, 870985728, 870988032, 870990336, 870997248, 886922496, 886924800, 892233216, 892235520, 892237824, 892240128, 913473792, 913483008, 934716672, 934718976, 934721280, 934728192, 950653440, 950655744, 955964160, 955966464, 955968768, 955971072, 977204736, 977213952, 998447616, 998449920, 998452224, 998459136, 1014384384, 1014386688, 1019695104, 1019697408, 1019699712, 1019702016, 1040935680, 1040944896, 1062178560, 1062180864, 1062183168, 1062190080, 1078115328, 1078117632, 1083426048, 1083428352, 1083430656, 1083432960, 1104666624, 1104675840, 1125909504, 1125911808, 1125914112, 1125921024, 1141846272, 1141848576, 1147156992, 1147159296, 1147161600, 1147163904, 1168397568, 1168406784, 1189640448, 1189642752, 1189645056, 1189651968, 1205577216, 1205579520, 1210887936, 1210890240, 1210892544, 1210894848, 1232128512, 1232137728, 1253371392, 1253373696, 1253376000, 1253382912, 1269308160, 1269310464, 1274618880, 1274621184, 1274623488, 1274625792, 1295859456, 1295868672, 1317102336, 1317104640, 1317106944, 1317113856, 1333039104, 1333041408, 1338349824, 1338352128, 1338354432, 1338356736, 1359590400, 1359599616, 1380833280, 1380835584, 1380837888, 1380844800, 1396770048, 1396772352, 1402080768, 1402083072, 1402085376, 1402087680, 1423321344, 1423330560, 1444564224, 1444566528, 1444568832, 1444575744, 1460500992, 1460503296, 1465811712, 1465814016, 1465816320, 1465818624, 1487052288, 1487061504, 1508295168, 1508297472, 1508299776, 1508306688, 1524231936, 1524234240, 1529542656, 1529544960, 1529547264, 1531906560]).cuda()

        # For GPT-3 Medium
        self.bin_edges = torch.tensor([0, 1024, 2048, 3072, 4197376, 4201472, 8395776, 8396800, 8397824, 8400896, 11546624, 11547648, 12596224, 12597248, 12598272, 12599296, 16793600, 16797696, 20992000, 20993024, 20994048, 20997120, 24142848, 24143872, 25192448, 25193472, 25194496, 25195520, 29389824, 29393920, 33588224, 33589248, 33590272, 33593344, 36739072, 36740096, 37788672, 37789696, 37790720, 37791744, 41986048, 41990144, 46184448, 46185472, 46186496, 46189568, 49335296, 49336320, 50384896, 50385920, 50386944, 50387968, 54582272, 54586368, 58780672, 58781696, 58782720, 58785792, 61931520, 61932544, 62981120, 62982144, 62983168, 62984192, 67178496, 67182592, 71376896, 71377920, 71378944, 71382016, 74527744, 74528768, 75577344, 75578368, 75579392, 75580416, 79774720, 79778816, 83973120, 83974144, 83975168, 83978240, 87123968, 87124992, 88173568, 88174592, 88175616, 88176640, 92370944, 92375040, 96569344, 96570368, 96571392, 96574464, 99720192, 99721216, 100769792, 100770816, 100771840, 100772864, 104967168, 104971264, 109165568, 109166592, 109167616, 109170688, 112316416, 112317440, 113366016, 113367040, 113368064, 113369088, 117563392, 117567488, 121761792, 121762816, 121763840, 121766912, 124912640, 124913664, 125962240, 125963264, 125964288, 125965312, 130159616, 130163712, 134358016, 134359040, 134360064, 134363136, 137508864, 137509888, 138558464, 138559488, 138560512, 138561536, 142755840, 142759936, 146954240, 146955264, 146956288, 146959360, 150105088, 150106112, 151154688, 151155712, 151156736, 151157760, 155352064, 155356160, 159550464, 159551488, 159552512, 159555584, 162701312, 162702336, 163750912, 163751936, 163752960, 163753984, 167948288, 167952384, 172146688, 172147712, 172148736, 172151808, 175297536, 175298560, 176347136, 176348160, 176349184, 176350208, 180544512, 180548608, 184742912, 184743936, 184744960, 184748032, 187893760, 187894784, 188943360, 188944384, 188945408, 188946432, 193140736, 193144832, 197339136, 197340160, 197341184, 197344256, 200489984, 200491008, 201539584, 201540608, 201541632, 201542656, 205736960, 205741056, 209935360, 209936384, 209937408, 209940480, 213086208, 213087232, 214135808, 214136832, 214137856, 214138880, 218333184, 218337280, 222531584, 222532608, 222533632, 222536704, 225682432, 225683456, 226732032, 226733056, 226734080, 226735104, 230929408, 230933504, 235127808, 235128832, 235129856, 235132928, 238278656, 238279680, 239328256, 239329280, 239330304, 239331328, 243525632, 243529728, 247724032, 247725056, 247726080, 247729152, 250874880, 250875904, 251924480, 251925504, 251926528, 251927552, 256121856, 256125952, 260320256, 260321280, 260322304, 260325376, 263471104, 263472128, 264520704, 264521728, 264522752, 264523776, 268718080, 268722176, 272916480, 272917504, 272918528, 272921600, 276067328, 276068352, 277116928, 277117952, 277118976, 277120000, 281314304, 281318400, 285512704, 285513728, 285514752, 285517824, 288663552, 288664576, 289713152, 289714176, 289715200, 289716224, 293910528, 293914624, 298108928, 298109952, 298110976, 298114048, 301259776, 301260800, 302309376, 302310400, 302311424, 303360000]).cuda()

        self.histogram_zeros = torch.zeros(len(self.bin_edges) - 1, dtype=torch.int64).cuda()

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.is_communication_outstanding = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.is_communication_outstanding
        ), 'Should not have multiple communication calls outstanding at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        if self.gradient_scaling_factor != 1.0:
            self.grad_data *= self.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        ### HANS: LOCAL GRADIENT CLIPPING ###
        # if self.ddp_config.local_clip_grad > 0.0:
        #     assert self.grad_data is not None
        #     norm = self.grad_data.norm(p=2)
        #     if not norm.isnan() and not norm.isinf():
        #         clip_coeff = self.ddp_config.local_clip_grad / (norm + 1.0e-6)
        #         if torch.distributed.get_rank() == 0:
        #             print("Max:", self.ddp_config.local_clip_grad, "norm", norm)
        #         if clip_coeff < 1.0:
        #             if torch.distributed.get_rank() == 0:
        #                 print("Clip:", clip_coeff)
        #             self.grad_data *= clip_coeff
        #             # print(self.grad_data)
        # # print(norm)

        # HANS: Additionals
        self.iteration += 1

        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:
            local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            self.communication_handle = torch.distributed._reduce_scatter_base(
                local_data_view,
                self.grad_data,
                op=reduce_op,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce,
            )
        else:
            ### HANS: TOPK, ONLY FOR TIMING ###
            # how_top = 0.01f
            # print("SHAPE:", self.grad_data.shape)
            # grad_abs = torch.abs(self.grad_data)
            # topk_val, topk_idx = torch.topk(grad_abs, k=int(how_top * self.grad_data.shape[0]), sorted=False)
            # dummy = torch.cat([topk_val, topk_idx])

            # self.communication_handle = torch.distributed.all_reduce(
            #     dummy,
            #     op=reduce_op,
            #     group=self.data_parallel_group,
            #     async_op=self.ddp_config.overlap_grad_reduce,
            # )
            ### END OF TOPK ###

            ### HANS: TO MEASURE COMMUNICATION TIME ###
            # if torch.distributed.get_rank() == 0:
                # start_time=time.time()
            ### MEASURING ENDS HERE ###

            ### HANS: FORCE MESSAGE TO BE SMALL ###
            # dummy = self.grad_data[0:1]
            # self.communication_handle = torch.distributed.all_reduce(
                # dummy,
                # op=reduce_op,
                # group=self.data_parallel_group,
                # async_op=self.ddp_config.overlap_grad_reduce,
            # )

            ### HANS: GRADIENT DUMPING ###
            # print("Dumping gradients at rank", torch.distributed.get_rank(), "...")
            # # gradname = "/home/lustre/NLP/Megatron-LM_clean/gradients/gpt1.7B_new/grad_iter50000_gpu" + str(torch.distributed.get_rank()) + "_bucket" + str(self.bucket_id)
            # gradname = "/home/lustre/NLP/Megatron-LM_clean/gradients/bertl_test/grad_iter10000_gpu" + str(torch.distributed.get_rank()) + "_bucket" + str(self.bucket_id+1)
            # # gradname = "/home/lustre/NLP/Megatron-LM_clean/gradients/bertl/grad_iter400000_gpu" + str(torch.distributed.get_rank())
            # # gradname = "/home/lustre/NLP/Megatron-LM_clean/gradients/gpt1.7B_fp32/grad_iter50000_gpu" + str(torch.distributed.get_rank()) + "_bucket" + str(self.bucket_id)

            # count_nan = torch.count_nonzero(torch.isnan(self.grad_data))
            # count_inf = torch.count_nonzero(torch.isinf(self.grad_data))

            # # print("Max:", torch.max(self.grad_data))
            # # print("Min:", torch.min(self.grad_data))
            # # print("NANs:", count_nan)
            # # print("INFs:", count_inf)

            # result = torch.histc(self.grad_data.float(), 500, min=-10, max=10) # HANS: histc does not work with more bins :(

            # hist_at_cpu = result.cpu().numpy()
            # with open(gradname, '
            # w') as f:
            #     np.savetxt(f, hist_at_cpu)
            # torch.cuda.synchronize()

            # if self.bucket_id >= 301:
            #     assert False # Stop program after dumping
            ### END OF GRADIENT DUMPING ###

            ### HANS: CHECK GRADIENT EXPLOSION ###
            # nan_dumpname = "/home/lustre/NLP/Megatron-LM_clean/dumps/nan-gpu" + str(torch.distributed.get_rank())
            # inf_dumpname = "/home/lustre/NLP/Megatron-LM_clean/dumps/inf-gpu" + str(torch.distributed.get_rank())
            # count_nan = torch.count_nonzero(torch.isnan(self.grad_data))
            # count_inf = torch.count_nonzero(torch.isinf(self.grad_data))
            
            # count_not_nan_cpu = count_nan.cpu().numpy()
            # count_not_inf_cpu = count_inf.cpu().numpy()

            # with open(nan_dumpname, 'a') as f:
            #     # np.savetxt(f, count_nan_cpu)
            #     print(count_nan_cpu, file=f)

            # with open(inf_dumpname, 'a') as f:
            #     # np.savetxt(f, count_inf_cpu)
            #     print(count_inf_cpu, file=f)

            # torch.cuda.synchronize()
            ### END OF CHECK GRADIENT EXPLOSION ###

            ## HANS: TOPK ###
            # prune_ratio = 0.5
            # top_ratio = 1 - prune_ratio

            # # Find threshold
            # k = int(top_ratio * self.grad_data.size()[0])
            # data_abs = torch.abs(self.grad_data)
            # temp = torch.topk(data_abs, k, largest=True)
            # threshold = temp.values[-1] # Select the last value as the threshold

            # # Create the mask, then zero out the small values
            # mask = (data_abs >= threshold).bool()
            # self.grad_data *= mask

            ### END OF TOPK ###

            # ### HANS: BOTTOMK ###
            # prune_ratio = 0.25
            # bottom_ratio = 1 - prune_ratio
            # bottom_ratio = prune_ratio

            # # Find threshold
            # k = int(bottom_ratio * self.grad_data.size()[0])
            # data_abs = torch.abs(self.grad_data)
            # temp = torch.topk(data_abs, k, largest=False)
            # threshold = temp.values[-1]

            # Dump histogram

            # if self.iteration == 1 or (self.iteration % 5000) == 0:
            # if (self.iteration % 10) == 0:
            # if True:
            # if False:
                # dumpname = "/home/lustre/NLP/Megatron-LM_clean/dump-bertl_75bot_grads/dump_iter" + str(300000) + "_dev_" + str(torch.distributed.get_rank())
                # dumpname = "/home/lustre/NLP/Megatron-LM_clean/dump-gpt-medium_25bot_grads/dump_iter" + str(500) + "_dev_" + str(torch.distributed.get_rank()) + "_bucket" + str(self.bucket_id)
                # dumpname = "/home/lustre/NLP/Megatron-LM_clean/dump-gpt-medium_75bot_grads/dump_iter" + str(self.iteration) + "_dev_" + str(torch.distributed.get_rank()) + "_bucket" + str(self.bucket_id)

                # bin_indices = torch.bucketize(temp.indices, self.bin_edges, right=True)
                # bin_indices -= 1
                # hist = torch.bincount(bin_indices)

                # hist_at_cpu = hist.cpu().numpy()

                # with open(dumpname, 'w') as f:
                #     np.savetxt(f, hist_at_cpu)

            # if self.bucket_id == 7:
            # assert False

            # # Create the mask, then zero out the small values
            # mask = (data_abs <= threshold).bool()
            # self.grad_data *= mask

            ### END OF BOTTOMK ###

            ### HANS: RANDOM PRUNING ###

            # For BERT.
            # Protect pooler.dense.weight, lm_head.dense.weight, self_attention.linear_proj.weight (multiple layers)

            # self.bucket_id != 1 and \
            # self.bucket_id != 8 and \
            # (self.bucket_id - 20) % 12 > 0 and \
            
            # if self.grad_data.size()[0] > 1e6 and \
            # self.bucket_id < 287:

            # if not (self.bucket_id >= 2 and self.bucket_id <= 3) and \
            # not (self.bucket_id >= 5 and self.bucket_id <= 8) and \
            # not (self.bucket_id >= 287 and self.bucket_id <= 300):

            # For GPT-3 Medium
            # THE ULTIMATE Combination for GPT-3 Medium
            # if self.grad_data.size()[0] > 1e6 and \
            # (self.bucket_id - 3) % 12 > 0 and \
            # (self.bucket_id - 11) % 12 > 0 and \
            # self.bucket_id < 278 and \
            # self.bucket_id > 290:

            # (self.bucket_id - 6)  % 12 != 0 and \
            # (self.bucket_id - 7)  % 12 != 0 and \
            # (self.bucket_id - 12)  % 12 != 0 and \
            # (self.bucket_id - 13)  % 12 != 0:

            if not (self.bucket_id >= 278 and self.bucket_id <= 290) and \
            (self.bucket_id - 2)  % 12 != 0 and \
            (self.bucket_id - 3)  % 12 != 0 and \
            (self.bucket_id - 10) % 12 != 0 and \
            (self.bucket_id - 11) % 12 != 0 and \
            (self.bucket_id > 1):

                ### CHOOSE THE RATIO ###
                # 1. Static ratio
                ratio = 0.5

                # 2. Dynamic ratio, ala Prof. John Kim
                # rand_ratio = torch.cuda.FloatTensor(1).uniform_(0.0, 1.0)
                # rand_ratio_int = int(rand_ratio * 8)
                # ratio = rand_ratio_int / 8

                offset = int(ratio * torch.distributed.get_world_size())

                rand = torch.cuda.FloatTensor(self.grad_data.size()[0]).uniform_(0.0, 1.0)

                lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
                hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

                lo_mask = (rand < lo_thres).bool()
                hi_mask = (rand >= hi_thres).bool()

                # HANS: For debugging
                # if torch.distributed.get_rank() == 2:
                    # print("Lo thres", lo_thres, ", hi thres", hi_thres)

                if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
                    mask = lo_mask * hi_mask # And
                else:
                    mask = lo_mask + hi_mask # Or

                # HANS: For debugging
                # if torch.distributed.get_rank() == 2:
                # print("True ratio at", torch.distributed.get_rank(), "is", torch.count_nonzero(mask) / self.grad_data.size()[0])

                self.grad_data *= mask

            ### END OF RANDOM PRUNING ###
            
            ### THE ORIGINAL ALLREDUCE: UNCOMMENT THIS PART TO USE THE BASELINE ###
            self.communication_handle = torch.distributed.all_reduce(
                self.grad_data,
                op=reduce_op,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce,
            )
            ### END OF THE ORIGINAL ALLREDUCE ###

            ### HANS: TO MEASURE COMMUNICATION TIME ###
            # torch.distributed.barrier()
            # if torch.distributed.get_rank() == 0:
                # print(time.time() - start_time)
            ### MEASURING ENDS HERE ###

        if self.ddp_config.overlap_grad_reduce:
            self.is_communication_outstanding = True
        else:
            self.is_communication_outstanding = False

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.is_communication_outstanding, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()


class ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
    ):
        self.ddp_config = ddp_config

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.gradient_scaling_factor = gradient_scaling_factor
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0

        def _create_new_bucket(data_end_index: int) -> int:
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
            data_end_index = _pad_if_needed(data_end_index)
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))
            bucket_data_start_index = data_end_index
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()
            bucket_id += 1
            # Return the potentially padded data_end_index.
            return data_end_index
        
        # HANS: To create separate buckets for layer normalizations
        is_layernorm = False

        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel

            # HANS: For debugging
            # if torch.distributed.get_rank() == 0:
                # print(data_start_index)
                # print(param_to_name[param])


            def _does_param_require_new_bucket(param):
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (
                    getattr(param, "shared_embedding", False)
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            # if (
            #     bucket_size is not None
            #     and (data_end_index - bucket_data_start_index) >= bucket_size
            # ) or _does_param_require_new_bucket(param):
            #     data_end_index = _create_new_bucket(data_end_index)

            # HANS: Create a bucket for each parameter, for debugging
            if True:
                data_end_index = _create_new_bucket(data_end_index)

            # HANS: Create separate buckets for layer normalizations. All params before these will be grouped in the same bucket as these params.
            # if param_to_name[param].find("output_layer.bias") >= 0 or param_to_name[param].find("lm_head.dense.weight") >= 0 \
            # or param_to_name[param].find("linear_fc1.weight") >= 0 or param_to_name[param].find("linear_proj.weight") >= 0 \
            # or param_to_name[param].find("layer_norm.weight") >= 0 or param_to_name[param].find("layernorm.weight") >= 0:
            #     data_end_index = _create_new_bucket(data_end_index)

            # HANS: Create separate buckets for each layer.
            # if param_to_name[param].find("input_layernorm.weight") >= 0 or param_to_name[param].find("final_layernorm.weight") >= 0 \
            # or param_to_name[param].find("word_embeddings.weight") >= 0:
                # data_end_index = _create_new_bucket(data_end_index)

            # HANS: Create separate bucket for heavy layers
            # if param_to_name[param].find("dense.bias") >= 0 or param_to_name[param].find("dense.weight") >= 0 \
            # or param_to_name[param].find("linear_fc2.bias") >= 0 or param_to_name[param].find("linear_fc2.weight") >= 0 \
            # or param_to_name[param].find("linear_fc1.bias") >= 0 or param_to_name[param].find("linear_fc1.weight") >= 0 \
            # or param_to_name[param].find("linear_qkv.bias") >= 0 or param_to_name[param].find("linear_qkv.weight") >= 0 \
            # or param_to_name[param].find("linear_proj.bias") >= 0 or param_to_name[param].find("linear_proj.weight") >= 0 \
            # or param_to_name[param].find("tokentype_embeddings.weight") >= 0 or param_to_name[param].find("position_embeddings.weight") >= 0:
            #     data_end_index = _create_new_bucket(data_end_index)

            # if param_to_name[param].find("layernorm.weight") >= 0:
            #     if is_layernorm is False:
            #         data_end_index = _create_new_bucket(data_end_index)
            #     is_layernorm = True
            # else:
            #     if is_layernorm is True:
            #         data_end_index = _create_new_bucket(data_end_index)
            #     is_layernorm = False

            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            data_end_index = _create_new_bucket(data_end_index)

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(
            self.numel,
            dtype=self.grad_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD
            )
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params=bucket_params,
                start_index=bucket_data_start_index,
                end_index=bucket_data_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )

        # Log buckets for all PP stages.
        log_strs = []
        log_strs.append(
            f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
        )
        for index, bucket in enumerate(self.buckets):
            numel = 0
            for param in bucket.params:
                numel += param.data.nelement()
            log_strs.append(f'Params for bucket {index+0} ({numel} elements):')
            for param in bucket.params:
                log_strs.append(f'\t{param_to_name[param]}')
        log_on_each_pipeline_stage(logger, logging.INFO, '\n'.join(log_strs))

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = Bucket(
            ddp_config=self.ddp_config,
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)

    # HANS: Additionals for LOCAL GRADIENT CLIPPING
    def local_grad_norm(self) -> None:

        if self.ddp_config.local_clip_grad > 0.0:
            assert self.grad_data is not None
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan()
            assert not norm.isinf()

            if not norm.isnan() and not norm.isinf():
                clip_coeff = self.ddp_config.local_clip_grad / (norm + 1.0e-6)
                # if torch.distributed.get_rank() == 0:
                    # print("Max:", self.ddp_config.local_clip_grad, "norm", norm)
                if clip_coeff < 1.0:
                    # if torch.distributed.get_rank() == 0:
                        # print("Clip:", clip_coeff)
                    self.grad_data *= clip_coeff