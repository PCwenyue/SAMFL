#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch.nn.functional as F
# from attention import Self_Attn

try:
	from correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './pwc_part/correlation'); import correlation # you should consider upgrading python
# end

##########################################################

# assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

# torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################


##########################################################

Backward_tensorGrid = {}
Backward_tensorPartial = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	if str(tensorFlow.size()) not in Backward_tensorPartial:
		Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
	tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

	tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

	tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

	return tensorOutput[:, :-1, :, :] * tensorMask
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Self_Attn(nn.Module):
			""" Self attention Layer"""
			def __init__(self,in_dim,activation=None):
				super(Self_Attn,self).__init__()
				self.chanel_in = in_dim
				self.activation = activation
				
				self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
				self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
				self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
				self.gamma = nn.Parameter(torch.zeros(1))

				self.softmax  = nn.Softmax(dim=-1) #
			def forward(self,x):
				"""
					inputs :
						x : input feature maps( B X C X W X H)
					returns :
						out : self attention value + input feature 
						attention: B X N X N (N is Width*Height)
				"""
				m_batchsize,C,width ,height = x.size()
				proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
				proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
				energy =  torch.bmm(proj_query,proj_key) # transpose check
				attention = self.softmax(energy) # BX (N) X (N) 
				proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

				out = torch.bmm(proj_value,attention.permute(0,2,1) )
				out = out.view(m_batchsize,C,width,height)
				
				out = self.gamma*out + x
				# return out,attention
				return out

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()


				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(16),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(32),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(64),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(96),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(128),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					#Self_Attn(196),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2+1, 81 + 64 + 2 + 2+1, 81 + 96 + 2 + 2+1, 81 + 128 + 2 + 2+1, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2+1, 81 + 64 + 2 + 2+1, 81 + 96 + 2 + 2+1, 81 + 128 + 2 + 2+1, 81, None ][intLevel + 0]

				intFeatExtractor = [None,16,32,64,96,128,196][intLevel]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1))
				
				self.occDector = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intFeatExtractor, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
					torch.nn.Sigmoid()
					)
				
			# end

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorSecond = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)
					tensorWarp = tensorSecond-tensorFirst
					occMap = self.occDector(tensorWarp)


					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat,occMap ], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32+1, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)
			# end
		# end
		class pool_refiner(torch.nn.Module):
			def __init__(self):
				super(pool_refiner,self).__init__()
				self.refine11 = torch.nn.Sequential(nn.Conv2d(568, 300, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.refine12 = torch.nn.Sequential(nn.Conv2d(300, 300, kernel_size=3,stride=1,padding=1),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))		

				self.refine21 = torch.nn.Sequential(nn.Conv2d(308, 400, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))			
				self.refine22 = torch.nn.Conv2d(400, 2, kernel_size=3,stride=1,padding=1)
													


				self.conv1010 = torch.nn.Sequential(nn.Conv2d(300, 2, kernel_size=1,stride=1,padding=0),
								torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv1020 = torch.nn.Sequential(nn.Conv2d(300, 2, kernel_size=1,stride=1,padding=0),
								torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv1030 = torch.nn.Sequential(nn.Conv2d(300, 2, kernel_size=1,stride=1,padding=0),
								torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv1040 = torch.nn.Sequential(nn.Conv2d(300, 2, kernel_size=1,stride=1,padding=0),
								torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

				self.conv10100 = torch.nn.Sequential(nn.Conv2d(300+529, 300, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv10200 = torch.nn.Sequential(nn.Conv2d(300+661+1, 300, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv10300 = torch.nn.Sequential(nn.Conv2d(300+629+1, 300, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
				self.conv10400 = torch.nn.Sequential(nn.Conv2d(300+597+1, 300, kernel_size=1,stride=1,padding=0),
													torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))


				self.att_1010 = Self_Attn(300)
				self.att_1020 = Self_Attn(300)
				self.att_1030 = Self_Attn(300)
				self.att_1040 = Self_Attn(300)

				self.upsample = F.upsample_nearest


			def forward(self,tensorInput,feature):  #0:torch.Size([8, 529, 6, 8]),1:[8, 662, 12, 16],2:torch.Size([8, 630, 24, 32]),3:torch.Size([8, 598, 48, 64]),4:[8, 566, 96, 128]
				_,_,h,w = tensorInput.size()
				shape_out = (h,w)

				tensorInput = torch.cat([tensorInput,feature[-1]],1)##568
				tensorInput = self.refine11(tensorInput)
				tensorInput = self.refine12(tensorInput)

    			


				x101 = F.avg_pool2d(tensorInput, 16)
				x102 = F.avg_pool2d(tensorInput, 8)
				x103 = F.avg_pool2d(tensorInput, 4)
				x104 = F.avg_pool2d(tensorInput, 2)

				x1010 = self.upsample(self.conv1010(self.att_1010(self.conv10100(torch.cat([x101,feature[0]],1)))),size=shape_out)
				x1020 = self.upsample(self.conv1020(self.att_1020(self.conv10200(torch.cat([x102,feature[1]],1)))),size=shape_out)
				x1030 = self.upsample(self.conv1030(self.att_1030(self.conv10300(torch.cat([x103,feature[2]],1)))),size=shape_out)
				x1040 = self.upsample(self.conv1040(self.att_1040(self.conv10400(torch.cat([x104,feature[3]],1)))),size=shape_out)

				flowFeature = torch.cat((x1010, x1020, x1030, x1040, tensorInput ), 1)
				flow = self.refine21(flowFeature)
				flow = self.refine22(flow)

				return flow




		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()
		
		self.poolRefiner = pool_refiner()

		# self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end
		for m in self.modules():
			classname = m.__class__.__name__
			if classname.find('Self_Attn') != -1:
				kaiming_normal_(m.key_conv.weight, 0.1)
				kaiming_normal_(m.query_conv.weight, 0.1)
				kaiming_normal_(m.value_conv.weight, 0.1)
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				try:
					kaiming_normal_(m.weight, 0.1)
				except:
					pass
				if m.bias is not None:
					constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				constant_(m.weight, 1)
				constant_(m.bias, 0)

	def forward(self, tensorFirst, tensorSecond):
		feature = []
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate6 = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate5 = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate6)
		objectEstimate4 = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate5)
		objectEstimate3 = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate4)
		objectEstimate2 = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate3)

		feature = [objectEstimate6['tensorFeat'],objectEstimate5['tensorFeat'],objectEstimate4['tensorFeat'],
					objectEstimate3['tensorFeat'],objectEstimate2['tensorFeat']]

		objectEstimate2 = objectEstimate2['tensorFlow'] + self.moduleRefiner(objectEstimate2['tensorFeat'])+\
							 self.poolRefiner(objectEstimate2['tensorFlow'],feature)
		
		# objectEstimate2 = objectEstimate2['tensorFlow'] + self.moduleRefiner(objectEstimate2['tensorFeat'])


		
		if self.training:
			return [objectEstimate2, objectEstimate3['tensorFlow'],objectEstimate4['tensorFlow'],
			objectEstimate5['tensorFlow'], objectEstimate6['tensorFlow']]
		else:
			return objectEstimate2,
		
	# end
# end

# moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = 20.0 * torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################

# if __name__ == '__main__':
# 	tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
# 	tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

# 	tensorOutput = estimate(tensorFirst, tensorSecond)

# 	objectOutput = open(arguments_strOut, 'wb')

# 	numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
# 	numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
# 	numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

# 	objectOutput.close()
# # end