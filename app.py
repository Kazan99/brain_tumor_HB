import os;
import tensorflow as tf;
import numpy as np;
from keras.preprocessing import image;
from PIL import Image;
import cv2;
from keras.models import load_model;
from flask import Flask, request, render_template;
from werkzeug.utils import secure_filename;

app = Flask(__name__);

# model = load_model('BrainTumor10Epochs.h5');
model = load_model('BrainTumor10EpochsCategorical.h5')

print('Model loaded. Check http://127.0.0.1:5001/');

# def get_classRadio1(info):

def get_className(classNo, info, info2, info3, info4):
    # if classNo == 0:
    #     return ' ko co'
    # if classNo == 1:
    #     return 'co'
    if classNo == 0:
        if info == "no" and info2 == "no" and info3 == "no" and info4 == "no" :
            thong_tin = ("Вы в польностью здоровы");

        elif info == "yes" and info2 == "no" and info3 == "no" and info4 == "no" :
            thong_tin = ('В МРТ не обнаружено изображение опухоли в головном мозге.'
                        ' Однако, сильные головные боли часто возникают у примерно половины пациентов с опухолями головного мозга.'
                        ' Боль обычно усиливается рано утром или поздно вечером, характеризуется постоянным характером и ежедневным повторением,'
                        ' а также увеличивается как по интенсивности, так и по продолжительности. Если у вас сильная боль, важно обратиться в медицинский центр для проведения обследования.');
        
        elif info == "no" and info2 == "yes" and info3 == "no" and info4 == "no":
            thong_tin = ('В МРТ не обнаружено изображение опухоли в головном мозге. Однако рвота и тошнота также признак того, что у вас может начаться опухоль головного мозга.' 
                        ' Сначала признаки не были ясными. У небольшого числа пациентов были диагностированы простые симптомы рвоты, и подозревалось наличие проблем с пищеварением.' 
                        ' Пока не были проведены анализы и обследования, результаты оставались неясными. Более четко стало видно, что это была рвота, вызванная опухолью головного мозга.');
        
        elif info == "no" and info2 == "no" and info3 == "yes" and info4 == "no" :
            thong_tin = ('В МРТ не обнаружено изображение опухоли в головном мозге. Однако потеря зрения также является признаком того, что у вас может начаться опухоль головного мозга.' 
                        'В начале признаки не были ясными, и у небольшого числа пациентов были диагностированы аномалии рефракции. Только после проведения тестов и получения более четких' 
                        'результатов стало понятно, что это опухоли головного мозга. При подозрении на повышение внутричерепного давления для подтверждения следует провести офтальмоскопию,' 
                        'так как после стадии отека диска зрительного нерва он переходит в атрофию сосочков, что может привести к слепоте.');
        
        elif info == "no" and info2 == "no" and info3 == "no" and info4 == "yes" :
            thong_tin = ('В МРТ не обнаружено изображение опухоли в головном мозге. Однако, раздражительность, усталость, стресс, возбуждение, плохая концентрация,' 
                        'частая сонливость или постоянная сонливость также являются симптомами, на которые следует обратить внимание. Поэтому вам всё равно стоит регулярно проходить' 
                        'медицинские осмотры, чтобы поддерживать здоровое состояние организма.');

        # if info == "yes" or info2 == "yes" or info3 == "yes" or info4 == "yes":
        else:
            thong_tin = ('В МРТ не обнаружено изображение опухоли в головном мозге. Однако симптомы указывают на возможное наличие опухоли, которую МРТ не выявляет.' 
                        ' Поэтому немедленно обратитесь в медицинский центр для более точной диагностики и лечения, чтобы поддерживать здоровье.');              
    if classNo == 1:
        if info == "yes" or info2 == "yes" or info3 == "yes" or info4 == "yes":
            thong_tin = ('В МРТ обнаружено изображение опухоли в головном мозге. Симптомы также указывают на наличие опухоли, поэтому немедленно обратитесь в медицинский' 
                        'центр для более точной диагностики и лечения, чтобы поддерживать здоровье.');
    return thong_tin;
    
# upload anh 
def getResult(img):
    image = cv2.imread(img);
    image = Image.fromarray(image, 'RGB');
    image = image.resize((64, 64));
    image = np.array(image);
    input_img = np.expand_dims(image, axis=0);

    result = np.argmax(model.predict(input_img), axis= 1);
    return result;


#route home
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html');
#route BrainTumor
@app.route('/new', methods=['GET'])
def index():
    return render_template('index.html');

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file'];
        e1 = request.form['benh'];
        e2 = request.form['benh2'];
        e3 = request.form['benh3'];
        e4 = request.form['benh4'];
        
        basepath = os.path.dirname(__file__);
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        );
        f.save(file_path);
        value = getResult(file_path);
        result = get_className(value, e1, e2, e3, e4);
        # result = get_className(value);

        return result;
    return None;

if __name__ == '__main__':
    app.run(debug=True);
