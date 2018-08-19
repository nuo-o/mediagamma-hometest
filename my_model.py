from my_packages import *


class MyAbstractModel():
    model = None

    def __init__(self):
        pass

    @abstractmethod
    def make_feat_pipeline(self, df, param):
        pass

    @abstractmethod
    def train(self, x, y, param):
        pass

    def compute_auc(self, pred, y, draw=False):
        fpr, tpr, _ = roc_curve(y, pred, pos_label=1)
        AUC = "%.4f" % auc(fpr, tpr)

        # if draw:
        #     self.FPR = fpr
        #     self.TPR = tpr
        #     self.AUC = AUC
        #     title = 'ROC Curve, AUC = ' + str(AUC)
        #
        #     with plt.style.context(('ggplot')):
        #         fig, ax = plt.subplots()
        #         ax.plot(fpr, tpr, "#000099", label='ROC curve')
        #         ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        #         plt.xlim([0.0, 1.0])
        #         plt.ylim([0.0, 1.05])
        #         plt.xlabel('False Positive Rate')
        #         plt.ylabel('True Positive Rate')
        #         plt.legend(loc='lower right')
        #         plt.title(title)
        return AUC

    @abstractmethod
    def predict(self, x):
        pass

    def dump_model(self, saved_name):
        if self.model != None:
            joblib.dump(self.model, saved_name)
        else:
            raise ValueError('no available model to save')

    def load_model(self, saved_name):
        self.model = joblib.load(saved_name)
        return self.model


class MyXgbModel(MyAbstractModel):

    def make_feat_pipeline(self, df, param):
        if param['do_sample']:
            df = down_sample_train(df, n=param['sample_ratio'])
  
        df = add_date_features(df, 'click_time')
        #df = add_groupby_features(df)
        df = df.drop(['click_time', 'device'], axis=1)
        
        return df

    def train(self, x, y, param):
        self.model = XGBClassifier(**param)
        self.model.fit(x, y)
        dMatrix = xgb.DMatrix(x, label=y)
        cv = xgb.cv(self.model.get_xgb_params(), dMatrix, nfold=5, early_stopping_rounds=10, metrics=['auc'])
        
        feat_imp = self.model.feature_importances_
        
        return cv.iloc[-1]['test-auc-mean'], feat_imp

    def predict(self, x):
        return self.model.predict_proba(x)[:, 1]


