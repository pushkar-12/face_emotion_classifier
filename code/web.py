import os,cv2,random,string
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory,jsonify,flash
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        # convert string of image data to uint8
        nparr = np.fromstring(request.data, np.uint8)

        #check if the api is being called through code or through web-form
        if(not(nparr.size==0)):

            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            random_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
            random_name=random_name+'.jpeg'
            random_name = os.path.join(app.config['UPLOAD_FOLDER'], random_name)

            cv2.imwrite(random_name, img)
            import image_emotion_color_demo
            response=jsonify(image_emotion_color_demo.process(random_name))
            try:
                os.remove(random_name)
            except OSError:
                pass
            return response

        else:

            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                x = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                file.save(x)

                import image_emotion_color_demo
                response=jsonify(image_emotion_color_demo.process(x))
                try:
                    os.remove(x)
                except OSError:
                    pass


                return response

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


app.run(host="0.0.0.0", port="8080")
