from config import Config
import numpy as np
from sklearn.linear_model import Ridge

class ThresholdTrainModel():

	def __init__(self, num_classes=Config.num_classes, l2=Config.classifier_l2_lambda):
		self.l2 = l2
		self.logits = None
		self.targets = None
		self.thresholds = np.full([num_classes], fill_value=0.5)  # initialize threshold to 0.5
		self.ridge = None
		self.POSSIBLE_THRESHOLDS = np.arange(0, 1, 0.01)

	def add_data(self, logits, targets):
		'''
		logits / targets are shape [batch_size, num_classes]
		'''
		# print('add_data')
		if self.logits is None:
			self.logits = logits
			self.targets = targets
		else:
			np.concatenate((self.logits, logits), axis=0)
			np.concatenate((self.targets, targets), axis=0)
		# print 'self.logits: %s' % (self.logits)
		# print 'self.targets: %s' % (self.targets)


	def clear_data(self):
		# print('clear_data')
		self.logits = self.targets = None


	def determine_threshold(self):
		# print('determine_threshold')
		if self.logits is None:
			raise 'No logits provided'
		
		# print('Determining threshold...')
		num_samples = self.logits.shape[0]
		num_classes = self.logits.shape[1]

		# logistics = self.calculate_logistics(self.logits)
		# print('logistics: %s' % (logistics))

		for i in xrange(num_classes):
			f_score_per_threshold = np.zeros_like(self.POSSIBLE_THRESHOLDS)
			for j, t in enumerate(self.POSSIBLE_THRESHOLDS):
				predicted_labels = self.predict_labels(self.logits[:,i], t)
				targets_arr = np.array([target[i] for target in self.targets])

				f_score_per_threshold[j] = self.calculate_F_score(predicted_labels, targets_arr)[0]
				# print('For threshold %f, f_score is: %f' % (t, f_score_per_threshold[j],))
			self.thresholds[i] = self.POSSIBLE_THRESHOLDS[np.argmax(f_score_per_threshold)]

		# print('Thresholds determined: %s' % (self.thresholds))
		return self.thresholds[:]

    
	def predict_labels(self, logistics, threshold):
		# print('predict_labels')
		return np.where(logistics > threshold, np.ones(logistics.shape), np.zeros(logistics.shape))


	def train_regression(self):
		# print('train_regression')
		self.ridge = Ridge(alpha=self.l2)
		self.ridge.fit(self.logits, np.tile(self.thresholds, (len(self.logits), 1)))


	def calculate_thresholds(self, logits):
		# print('calculate_thresholds')
		return self.ridge.predict(logits)


	def calculate_F_score(self, predicted_labels, targets):
		'''
		calcualtes F0.5 score for samples

		predicted_labels / targets are shape [batch_size] or [batch_size, num_classes]
		'''
		# print('calculate_F_score')

		precision = np.mean([np.divide(np.sum(np.logical_and(predicted_labels[i], targets[i])), np.sum(predicted_labels[i])) for i in xrange(len(predicted_labels))])
		recall = np.mean([np.divide(np.sum(np.logical_and(predicted_labels[i], targets[i])), np.sum(targets[i])) for i in xrange(len(predicted_labels))])

		f_point_five = 1.25 * precision * recall / (0.25 * precision + recall)

		return f_point_five, precision, recall