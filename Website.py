J = "Jahe"
K = "Kunyit"
L = "Lengkuas"
T = "Temulawak"
B = "Beranda"
P = "Passfoto"
A = "Bantuan"
I = "Instagram"
F = "Facebook"
Tw = "Twitter"

SampleA = 'static/A.jpg'
SampleJ = 'static/Jahe.jpeg'
SampleK = 'static/Kunyit.jpeg'
SampleL = 'static/Lengkuas.jpeg'
SampleT = 'static/Temulawak.jpeg'
SampleB = 'static/B.jpg'
SampleP = 'static/passpoto.jpg'
SampleI = 'static/ig.png'
SampleF = 'static/fb.png'
SampleTw = 'static/Tw.jpeg'
UPLOAD_FOLDER = 'static/uploads'

# Allowed Files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}



import os
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pathlib
from datetime import datetime, timedelta
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from jcopml.tuning import random_search_params as rsp
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd 
import argparse
import matplotlib.pyplot as plt
from feature_extraction import feature_extraction
from skimage import io, color, img_as_ubyte
from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte



#Create the website object

app = Flask(__name__)

def load_model_from_file():
	#Set up the machine learning session
	mySession = tf.Session()
	set_session(mySession)
	myModel = load_model('modelklasifikasi.pkl')
	myGraph = tf.get_default_graph()
	return (mySession,myModel,myGraph)
	
#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
	
@app.route('/')
def entry_page():
	return render_template('Beranda.html',myB=B,mySampleB=SampleB)

@app.route('/Bantuan')
def halaman2():
	return render_template('Bantuan.html', myA=A,myI=I,myF=F,myTw=Tw,mySampleA=SampleA,mySampleI=SampleI,mySampleF=SampleF,mySampleTw=SampleTw)

@app.route('/TentangAplikasi')
def halaman3():
	return render_template('TentangAplikasi.html',MyP=P,mySampleP=SampleP)

#Define the view for the top level page
@app.route('/Pendeteksian', methods=['GET', 'POST'])
def upload_file():
	#Initial webpage Load
	if request.method == 'GET' :
		return render_template('index.html',myJ=J,myK=K,myL=L,myT=T,mySampleJ=SampleJ,mySampleK=SampleK,mySampleL=SampleL,mySampleT=SampleT)
	else: #if request.method == 'POST':
		#check if the post request has the file part
		if 'file' not in request.files:
			flash('No File Part')
			return redirect(request.url)
		file = request.files['file']
		#if user does not select file, browser may also
		#submit an empty part without file name
		if file.filename == '':
			flash('No Selected file')
			return redirect(request.url)
		# If it doesn`t look like an image file
		if not allowed_file(file.filename):
			flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
			return redirect(request.url)
		#when the user uploads a file with good parameters
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('uploaded_file', filename=filename))
			
			
def auto_remove(filename):
    import pathlib
    import os
    from datetime import datetime, timedelta
    image_upload_dir =  'static/uploads'
    image_upload_list = os.listdir(image_upload_dir)
    datetime_now = datetime.now()

    for image in image_upload_list:
        fname = pathlib.Path(image)
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        if datetime_now + timedelta(minutes = 1) < mtime:
            os.remove(image_upload_dir + '/' + image)

			

@app.route('/Pendeteksian/Uploads/<filename>')
def uploaded_file(filename):
	test_image =(UPLOAD_FOLDER+"/"+filename)
	#test_image = image.img_to_array(test_image)
	#test_image = np.expand_dims(test_image, axis=0)
	pred_feature = feature_extraction(test_image)
	pred_df = pd.DataFrame([pred_feature])
	pred_df.columns = ['Hist', 'SobX', 'SobY', 'Cont', 'Corr', 'Ener', 'Homo']
	mySession = app.config['SESSION']
	myModel = app.config['MODEL']
	myGraph = app.config['GRAPH']
	with myGraph.as_default():
		set_session(mySession)
		result = myModel.predict(pred_df)
		image_src = "/"+UPLOAD_FOLDER+"/"+filename
		if result[0] == '0':
			print('Jahe')
			answer = "<div></div><div class='col-sm-4'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+J+" "+str(result[0])+"</h4><br></br><h4><i>Jahe bersifat anti-inflamasi dan anti-oksidatif yang bisa mengendalikan proses penuaan. Manfaat jahe lainnya, tanaman herbal ini juga memiliki potensi antimikroba yang dapat membantu dalam mengobati penyakit menular. Bahkan, manfaat jahe disebut-sebut dapat mencegah berbagai kanker</i></h4></div>"
		elif result[0] == '1':
			print('Kunyit')
			answer = "<div></div><div class='col-sm-4'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+K+" "+str(result[0])+"</h4><br></br><h4><i>Kurkumin kunyit berfungsi menekan respon peradangan pada sel tubuh, temasuk sel pankreas, lemak, dan otot. Reaksi ini dapat membantu mengurangi resistensi insulin, menurunkan kadar gula darah, dan kolesterol serta gangguan metabolik lainnya akibat berat badan berlebih</i></h4></div>"
		elif result[0] == '2':
			print('Lengkuas')
			answer = "<div></div><div class='col-sm-4'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+L+" "+str(result[0])+"</h4><br></br><h4><i>Lengkuas kaya akan vitamin C yang dapat berperan dalam peremajaan sel-sel kulit dan menjaga kulit akan tampak lebih muda dari penuaan. Selain itu, lengkuas juga dapat digunakan untuk menyembuhkan penyakit kulit seperti panu, luka bakar, gatal-gatal, hingga alergi</i></h4></div>"
		elif result[0] == '3':
			print('Temulawak')
			answer = "<div></div><div class='col-sm-4'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+T+" "+str(result[0])+"</h4><br></br><h4><i>Temulawak merangsang produksi empedu di kantung empedu, sehingga membantu meningkatkan fungsi pencernaan. Dengan rutin mengonsumsinya, maka berbagai masalah pencernaan bisa teratasi termasuk kembung, gas dan dispepsia. Rempah juga dapat bermanfaat bagi pengidap kondisi peradangan seperti kolitis ulserativa.</i></h4></div>"
		results.append(answer)
	return render_template('index.html',myJ=J,myK=K,myL=L,myT=T,mySampleJ=SampleJ,mySampleK=SampleK,mySampleL=SampleL,mySampleT=SampleT,len=len(results),results=results)
	




			
def main():
	(mySession,myModel,myGraph) = load_model_from_file()
	app.config['SECRET_KEY'] = 'super secret key'
	
	app.config['SESSION'] = mySession
	app.config ['MODEL'] = myModel
	app.config['GRAPH'] = myGraph
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit
	app.run(debug=True, host="0.0.0.0", port=80)
	
	#Create a running list of result
	
results = []
	
#Launch Everything
main()
