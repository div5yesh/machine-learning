#%%
import numpy as np
from sklearn.metrics import classification_report, log_loss, f1_score, accuracy_score


#%%
def get_correct(A, B):
    return np.array((A, A, A, B, B, A, B))

def get_pred(A, B):
    return np.array((A, A, B, A, B, B, A))

#%%
#3.b.i: Accuracy
print(accuracy_score(get_correct("A", "B"), get_pred("A", "B")))

#%%
print("none", f1_score(get_correct(1, 0), get_pred(1, 0), average=None))
print("labels", f1_score(get_correct(1, 0), get_pred(1, 0), average="binary", labels=[1,0]))
print("binary", f1_score(get_correct(1, 0), get_pred(1, 0), average="binary"))
print("micro", f1_score(get_correct(1, 0), get_pred(1, 0), average="micro"))
print("macro", f1_score(get_correct(1, 0), get_pred(1, 0), average="macro"))
# print("samples", f1_score(get_correct(1, 0), get_pred(1, 0), average="samples"))
print("weighted", f1_score(get_correct(1, 0), get_pred(1, 0), average="weighted"))
print("pos_label", f1_score(get_correct(1, 0), get_pred(1, 0), average="binary", pos_label=0))
print("labels", f1_score(get_correct("foo", "bar"), get_pred("foo", "bar"), average=None, labels=["foo", "bar"]))
print("labels", f1_score(get_correct("foo", "bar"), get_pred("foo", "bar"), average="micro", labels=["foo", "bar"]))

#%%
print(classification_report(get_correct(1, 0), get_pred(1, 0)))
print(log_loss(["spam", "ham", "ham", "spam"],  
      [[.1, .9], [.9, .1], [.8, .2], [.35, .65]], normalize=True))

#%%
