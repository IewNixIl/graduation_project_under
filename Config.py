from torchvision import transforms as T
from torch.utils.data import DataLoader

class config:



    #'''
    path='D:\\m1474000'#所有数据的总文件夹路径
    path_devision='D:\\Codes\\Graduation\\train_test_list'#存训练数据和测试数据划分的文件路径
    path_labels='D:\\labels\\labels'#存手动标记的标签位置
    path_labels_test='D:\\labels\\test'#测试集
    path_labels_val='D:\\labels\\val'#用于在训练过程中进行验证

    path_acc_record='D:\\codes\\Graduation\\RECORD\\accu'

    path_label_train=''
    path_mask_train=''

    path_label_low_save='F:\\label_test\\labels_low'
    path_predict_save='G:\\Graguation\\test_result\\1ite\\predict_1_model3'
    path_pr_save='G:\\M12\\pr\\s1+s2'
    path_img_save='G:\\Graguation\\test_result\\img'
    path_ndwi_save='F:\\label_test\\labels_ndwi'
    #'''



    #model35 thre:0.31 ; p:0.9527 ; r:0.9324 ; fmeasure:0.9424 ; pa:0.9635 ; iou:0.8911
    #model36 thre:0.24 ; p:0.9560 ; r:0.9307 ; fmeasure:0.9432 ; pa:0.9641 ; iou:0.8925
    #model37 thre:0.34 ; p:0.9615 ; r:0.9213 ; fmeasure:0.9410 ; pa:0.9630 ; iou:0.8885
    #model38 thre:0.52 ; p:0.9722 ; r:0.9342 ; fmeasure:0.9528 ; pa:0.9704 ; iou:0.9099
    #model39 thre:0.41 ; p:0.9640 ; r:0.9504 ; fmeasure:0.9572 ; pa:0.9727 ; iou:0.9178


    model='Unet'
    model_load_path=''#'F:\\Graduation\\project_v2\\MODELS\\model29'
    model_save_path='D:\\codes\\Graduation\\MODELS\\model3'
    model_save_path_checkpoints='D:\\codes\\Graduation\\MODELS\\checkpoints\\checkpoint.pkl'
    statistic_save_path='D:\\codes\\Graduation\\MODELS\\sta3'
    sub_dataset_train='train_sub3'


    input_band=12#输入的波段数
    use_denoise=False#SAR波段是否进行去噪平滑
    use_gpu=True
    batch_size=4
    num_workers=1
    learning_rate=0.0001
    max_epoch=1
    number_recording=100
    weight_decay=0.0005



    mean={2:[-11,-11],
            10:[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
            12:[-11,-11,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048]}
    std={2:[11,11],
            10:[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
            12:[11,11,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048]}
    transform_img=T.Compose([
        T.ToTensor(),
        #T.Normalize(mean=[-11,-11,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
        #std=[11,11,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048])#归一化到 [-1,1]
        T.Normalize(mean=mean[input_band],std=std[input_band])#归一化到 [-1,1]
        #T.Normalize(mean=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
        #std=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048])#归一化到 [-1,1]
    ])

    transform_label=T.Compose([
        T.ToTensor()
    ])
