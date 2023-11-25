import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def f1_eval(y_true, y_pred, thr=0.5):
    """
    Evaluate the F1 score for binary classification using a specified threshold.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        thr (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        float: F1 score.
    """

    f1 = f1_score(y_true, y_pred>0.5)
    return f1

class FocalLoss:
    """
    Focal Loss for binary classification by the XGBoost model.

    Args:
        alpha (float, optional): Focal loss balancing factor. Defaults to 0.25.
        gamma (float, optional): Focal loss focusing parameter. Defaults to 2.
    """

    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def robust_pow(self, num_base, num_pow):
        """
        Robust power function to avoid numerical instability.

        Args:
            num_base (np.ndarray): Base value.
            num_pow (np.ndarray): Power value.

        Returns:
            np.ndarray: Robust power of num_base and num_pow.
        """

        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    def focal_binary_object(self, dtrain, pred):
        """
        Computes focal loss and its derivatives.

        Args:
            dtrain (np.ndarray): Ground truth labels.
            pred (np.ndarray): Predicted probabilities.

        Returns:
            tuple:
                grad (np.ndarray): Gradient of focal loss.
                hess (np.ndarray): Hessian of focal loss.
        """

        gamma = self.gamma
        alpha = self.alpha
        y = dtrain
        p = 1.0 / (1.0 + np.exp(-pred))
        
        a1 = alpha*self.robust_pow(1-p, gamma)*(gamma*p*np.log(p)+p-1)
        a2 = (1-alpha)*self.robust_pow(p, gamma)*(gamma*(1-p)*np.log(1-p)-p)
        grad = y*a1 - (1-y)*a2

        gamma_2 = 2*gamma+1
        gamma_q = -gamma**2-gamma
        b1 = alpha*p*self.robust_pow(1-p, gamma)*(gamma_q*p*np.log(p) + gamma*np.log(p) - gamma_2*p+gamma_2)
        b2 = (1-alpha)*(1-p)*self.robust_pow(p, gamma)*(gamma_q*(1-p)*np.log(1-p)+gamma*np.log(1-p)-gamma_2*(1-p)+gamma_2)
        hess = y*b1+(1-y)*b2
        return grad, hess

def xgb_model(n_estimators=300, device="cuda:0"):
    """
    Creates an XGBoost model with custom Focal Loss objective and evaluation metric.

    Args:
        n_estimators (int, optional): Number of estimators. Defaults to 300.
        device (str, optional): Device to use for training. Defaults to "cuda:0" if available,
            otherwise "cpu".

    Returns:
        XGBClassifier: XGBoost model.
    """
    
    f = FocalLoss(alpha=0.8, gamma=3)
    model = XGBClassifier(
        objective=f.focal_binary_object,
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=0.5,
        max_depth=12,
        subsample=0.1,
        sampling_method="gradient_based",
        colsample_bytree=1,
        scale_pos_weight=1,
        enable_categorical=True, 
        device=device, 
        verbosity=1,
        eval_metric=f1_eval,
        importance_type="cover",
        radom_state=0,
    )
    return model