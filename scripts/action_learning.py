import pandas as pd
import cv2
import matplotlib.pyplot as plt
from grapeberry.src.prediction import detect_berry

# ===== import some csv extracted with the R code from Llorenc ===================================================
df = pd.read_csv('data/copy_from_database/images_grapevine22.csv')
df = df[(df['viewtypeid'] == 3) & df['imgangle'].isin([k*30 for k in range(12)])]  # 12 side image

# ==============================================================================================================

weights_path = 'data/grapevine/validation/backup5/config_1class_40000.weights'
config_path = 'data/grapevine/config_1class.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
model_det = cv2.dnn_DetectionModel(net)

# =========================================================================================================

selec = df[df['taskid'] > 5880]
folder = 'data/grapevine/grapevine22/'

for k_row, row in selec.sample(1).iterrows():
    path1 = 'V:/ARCH2022-05-18/{}/{}.png'.format(row['taskid'], row['imgguid'])
    #path2 = folder + 'ARCH2022-05-18_{}_{}_{}.png'.format(int(row.plantid), row.taskid, row.imgangle)

    img = cv2.imread(path1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred = detect_berry(img, model_det, confidence_threshold=0.2, nms_threshold=0.8)

    scores = pred['score'][pred['score'] > 0.7]
    if not scores.empty:
        a, b, c = len(pred), round(len(scores) / len(pred), 2), round(sum(scores > 0.99) / len(scores), 2)
        print(a, b, c)
        if a > 30 and c < 0.5:
            print(k_row, path1, 'low score')

    plt.figure(k_row)
    plt.imshow(img)
    for _, row in pred.iterrows():
        x, y, w, h, score = row
        if score > 0.7:
            col = 'y' if score > 0.95 else ('b' if score > 0.80 else 'r')
            plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], col + '-')
            plt.text(x, y - 5, str(round(score, 3)), color=col)










