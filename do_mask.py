import numpy as np
import torch
class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.mask_ori = []
        self.mask_shadow = []
        self.filter_small_index = {}
        self.filter_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
        else:
            pass
        return codebook, filter_index

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('filter_large_index', filter_large_index)
            print('filter_small_index', filter_small_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x
    
    # 计算每层参数tensor形状\参数量
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, args):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key] = args.pruning_rate #设置每层的剪枝率
        # different setting for  different architecture
        if args.arch == 'sres50-21':
            last_index = 207
            skip_list = [3, 42, 93, 168] # skip projection
            self.mask_shadow = [12, 24, 36, 51, 63, 75, 87, 102, 114, 126, 138, 150, 162, 177, 189, 201] # shadow conv
        if args.arch == 'sres50':
            last_index = 207
            skip_list = [3, 42, 93, 168] # skip projection
            self.mask_shadow = [12, 24, 36, 51, 63, 75, 87, 102, 114, 126, 138, 150, 162, 177, 189, 201] # shadow conv
        elif args.arch == 'sres101':
            last_index = 411
            skip_list = [3, 42, 93, 372] # skip projection
            self.mask_shadow = [12, 24, 36, 51, 63, 75, 87, 102, 114, 126, 138, 
                                150, 162, 174, 186, 198, 210, 222, 234, 246, 258, 270,
                                282, 294, 306, 318, 330, 342, 354, 366, 381, 393, 405] # shadow conv
        # to jump the last fc layer 只取conv跳过bn和fc
        self.mask_index = [x for x in range(0, last_index, 3)] # 所有conv
        
        # 跳过projection
        for x in skip_list:
            self.compress_rate[x] = 1
            self.mask_index.remove(x)
        print(self.mask_index) 
        
        for x in self.mask_shadow:
            self.mask_index.remove(x)
        print(self.mask_index)
        # 此时的self.mask_index不含projection、shadow,只有主干的conv index
   
 

    def init_mask(self, args):
        self.init_rate(args)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index: #只主干 conv  resnet50  49 
                # mask for norm criterion
                self.mat[index], self.filter_index[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
            if index in self.mask_shadow:   # resnet 50 16
                codebook_shadow = np.ones(self.model_length[index])
                if len(item.data.size()) == 4:
                    kernel_length = item.data.size()[1] * item.data.size()[2] * item.data.size()[3]
                    for x in range(0, len(self.filter_index[index-3])):
                        codebook_shadow[self.filter_index[index-3][x] * kernel_length: (self.filter_index[index-3][x] + 1) * kernel_length] = 0
                else:
                    pass
                self.mat[index] = self.convert2tensor(codebook_shadow)
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def act_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
            if index in self.mask_shadow:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")




    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
               
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
