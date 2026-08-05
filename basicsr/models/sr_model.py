import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import numpy as np
import numpy as np
from thop import profile, clever_format
import warnings
from thop.vision.basic_hooks import count_prelu
import torch.nn as nn



def pad_to_size(array, new_height, new_width):
    # 获取原始数组的尺寸
    old_height, old_width = array.shape

    # 计算需要填充的行数和列数
    top_pad = (new_height - old_height) // 2
    bottom_pad = new_height - old_height - top_pad
    left_pad = (new_width - old_width) // 2
    right_pad = new_width - old_width - left_pad

    # 使用 numpy 的 pad 函数进行填充
    padded_array = np.pad(array, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)

    return padded_array
def stitch_image(slices, original_width, original_height, slice_width, slice_height, overlap):
    """
    将切片拼接回原图
    """
    stitched_image = np.zeros((original_height, original_width), dtype=np.float32)
    count_map = np.zeros((original_height, original_width), dtype=np.float32)
    index = 0

    for y in range(0, original_height - slice_height + 1, slice_height - overlap):
        for x in range(0, original_width - slice_width + 1, slice_width - overlap):
            slice = slices[index]
            stitched_image[y:y + slice_height, x:x + slice_width] += slice
            count_map[y:y + slice_height, x:x + slice_width] += 1
            index += 1

    stitched_image /= count_map
    return stitched_image


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""



    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.flops_calculated = True # 标记是否已计算过FLOPs
        self.flops = 0.0  # 存储FLOPs结果
        self.params = 0.0  # 存储参数量结果

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
    def pre_process(self):
        # pad to multiplication of window_size
        return 0

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        # if 'hrshape' in data:
        #
        #
        # else:
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'noise'in data:
            self.noise=data['noise'].to(self.device)

            # self.noise_ima = data['noise'][1].to(self.device)
            self.gt=[self.gt,self.noise]


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if not self.flops_calculated:
            print("\n========== 开始计算模型FLOPs和参数量 ==========")
            # 选择用于统计的模型（优先ema模型）
            model = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
            model.eval()

            # 处理输入：如果是crop模式，取第一个patch；否则用完整输入
            input_tensor = self.lq
            # 适配crop模式（lq是patch列表时，取第一个patch）
            if isinstance(input_tensor, list) or (len(input_tensor.shape) > 4):
                input_tensor = input_tensor[0].unsqueeze(0) if len(input_tensor) > 0 else input_tensor

            try:
                # 统计FLOPs和参数量
                flops, params = profile(model, inputs=(input_tensor,), verbose=False)
                # 格式化输出（自动转换为GFLOPs/MFLOPs，Params转换为M/B）
                self.flops, self.params = clever_format([flops, params], "%.3f")
                print(f"模型FLOPs: {self.flops}")
                print(f"模型参数量: {params}")
                print(f"统计输入尺寸: {input_tensor.shape}")
                print("========== FLOPs计算完成 ==========\n")
                self.flops_calculated = True  # 标记为已计算，避免重复统计
            except Exception as e:
                print(f"FLOPs计算失败: {e}")
                self.flops_calculated = True  # 避免反复尝试

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):






        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            lr_patchs = val_data['lq']
            hr_patchs = val_data['gt']
            # noise_patchs = val_data['noise']

            gtlist = []
            lqlist = []

            outputlist = []
            hrshape = val_data['hrshape']
            lrshape = val_data['lrshape']

            #若crop
            # for i in range(len(lr_patchs)):
            #     lr = lr_patchs[i]
            #     hr = hr_patchs[i]
            #     # noise = noise_patchs[i]
            #     self.lq = lr.to(self.device)
            #     self.gt = hr.to(self.device)
            #     # self.nosie = noise.to(self.device)
            # # self.feed_data(val_data)
            #     self.test()
            #     visuals = self.get_current_visuals()
            #     gt = visuals['gt']
            #     lq = visuals['lq']
            #     out = visuals['result']
            #     gt = gt.squeeze().numpy()
            #     lq = lq.squeeze().numpy()
            #     out = out.squeeze().numpy()
            #     gtlist.append(gt)
            #     lqlist.append(lq)
            #     outputlist.append(out)
            #
            # results = stitch_image(outputlist, hrshape[0], hrshape[1], 256, 256, 64)
            # highq = stitch_image(gtlist, hrshape[0], hrshape[1], 256, 256, 64)
            # lowq = stitch_image(lqlist, lrshape[0], lrshape[1], 128, 128, 32)

            # results = stitch_image(outputlist, hrshape[0], hrshape[1], 512, 512, 128)
            # highq = stitch_image(gtlist, hrshape[0], hrshape[1], 512, 512, 128)
            # lowq = stitch_image(lqlist, lrshape[0], lrshape[1], 256, 256, 64)

            lr = lr_patchs
            hr = hr_patchs
            # noise = noise_patchs[i]
            self.lq = lr.to(self.device)
            self.gt = hr.to(self.device)
            # self.nosie = noise.to(self.device)
        # self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            highq = visuals['gt']
            lowq = visuals['lq']
            results = visuals['result']


            # results = (torch.from_numpy(results)).unsqueeze(0).unsqueeze(0)
            # highq = (torch.from_numpy(highq)).unsqueeze(0).unsqueeze(0)
            # lowq = (torch.from_numpy(lowq)).unsqueeze(0).unsqueeze(0)
            visuals['gt'] = highq
            visuals['lq'] = lowq
            visuals['result'] = results

            # visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            lq_img=tensor2img([visuals['lq']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_lq_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_lq_{current_iter}.png')
                    save_sr_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_sr_{current_iter}.png')
                    save_gt_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_gt_{current_iter}.png') if gt_img is not None else None
                else:
                    if self.opt['val']['suffix']:
                        save_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_lq_{self.opt["val"]["suffix"]}.png')
                        save_sr_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_sr_{self.opt["val"]["suffix"]}.png')
                        save_gt_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_gt_{self.opt["val"]["suffix"]}.png') \
                            if gt_img is not None else None
                    else:
                        save_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_lq_{self.opt["name"]}.png')
                        save_sr_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_sr_{self.opt["name"]}.png')
                        save_gt_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_gt_{self.opt["name"]}.png') \
                            if gt_img is not None else None
                imwrite(lq_img, save_lq_path)
                imwrite(sr_img, save_sr_path)
                if gt_img is not None:
                    imwrite(gt_img, save_gt_path)
                # if self.opt['is_train']:
                #     save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                #                              f'{img_name}_{current_iter}.png')
                # else:
                #     if self.opt['val']['suffix']:
                #         save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                #                                  f'{img_name}_{self.opt["val"]["suffix"]}.png')
                #     else:
                #         save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                #                                  f'{img_name}_{self.opt["name"]}.png')
                # imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()

        # 如果用了噪声分支
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            # 如果用了噪声分支
            out_dict['gt'] = self.gt[0].detach().cpu()
            # out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
