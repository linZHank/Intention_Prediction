import data_utils
import pdb


filedir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/dataset_config/travaltes_20180415"
height=224
width=224
imformat = "color"


train_data, train_labels = data_utils.get_train_data(filedir,
                                                     height,
                                                     width,
                                                     imformat)
# eval_data, eval_labels = data_utils.get_eval_data()
