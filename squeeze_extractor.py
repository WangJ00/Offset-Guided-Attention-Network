from __future__ import absolute_import, division
from torch import nn
from dataclasses import dataclass

@dataclass
class CopyFeatureInfo:
	index: int
	out_channels: int

class SqueezeExtractor(nn.Module):
	def __init__(self, model, features, fixed_feature=True):
		super(SqueezeExtractor, self).__init__()
		self.model = model
		self.features = features
		#print(fixed_feature)
		if fixed_feature:
			for param in self.features.parameters():
				param.requires_grad = False

	def get_copy_feature_info(self):
		"""
		Get [CopyFeatureInfo] when sampling such as maxpooling or conv2d which has the 2x2 stride.
		:return: list. [CopyFeatureInfo]
		"""
		raise NotImplementedError()

	def _get_last_conv2d_out_channels(self, features):
		for idx, m in reversed(list(enumerate(features.modules()))):
			if isinstance(m, nn.Conv2d):
				return int(m.out_channels)
		assert False