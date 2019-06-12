from flask import Flask, render_template, url_for
from flask_bootstrap import Bootstrap
from flask import request
from flask import jsonify

from boto.s3.connection import S3Connection
from boto.s3.key import Key

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import word_tokenize
import nltk

import pickle

BUCKET_NAME = 'crfmedical'
MODEL_FILE_NAME = 'crf_model.pkl'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME

#*****************************
#Remove later
#Install boto
#*****************************

app = Flask(__name__)
bootstrap = Bootstrap(app)

def get_model():
	global crf_model 
	
	conn = S3Connection()
	bucket = conn.get_bucket(BUCKET_NAME)
	key_obj = Key(bucket)
	key_obj.key = MODEL_FILE_NAME

	#with open("crf_model.pkl", "rb") as fp:
	#	crf_model = pickle.load(fp)

	contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)

	with open(MODEL_LOCAL_PATH, "rb") as fp:
		crf_model = pickle.load(fp)

	#print("Model loaded")

def word_2_features(sentence, word_index):
    
    word = sentence[word_index][0]
    pos_tag = sentence[word_index][1]
    
    # static features
    
    features = {
        'bias' : 1,
        'word[-3:]' : word[-3:],
        'word[-2:]' : word[-2:],
        
        'lower' : word.islower(),
        'upper' : word.isupper(),
        'digit' : word.isdigit(),
        'hyphen' : word.count('-'),
        'title' : word.istitle(),
        
        'postag' : pos_tag,
        'postag2' : pos_tag[:2],

        '#capitals' : sum(1 for c in word if c.isupper()),
    }
    
    # transition features
    
    # prev
    if word_index > 0:
        prev = sentence[word_index-1][0]
        prev_pos_tag = sentence[word_index-1][1]
         
        features.update({
            'prev_lower' : prev.islower(),
            'prev_upper' : prev.isupper(),
            'prev_title' : prev.istitle(),
            
            'prev_postag' : prev_pos_tag,
            'prev_postag2' : prev_pos_tag[:2]
        })    
    
    else:
        features['BOS'] = True
        
    if word_index < (len(sentence) - 1):
        nex = sentence[word_index+1][0]
        nex_pos_tag = sentence[word_index+1][1]
         
        features.update({
            'nex_lower' : nex.islower(),
            'nex_upper' : nex.isupper(),
            'nex_title' : nex.istitle(),
            
            'nex_postag' : nex_pos_tag,
            'nex_postag2' : nex_pos_tag[:2]
        })
        
    else:
        features['EOS'] = True
        
    return features

def get_features(sentence):
    return [word_2_features(sentence, i) for i in range(len(sentence))]

def preprocess(sentence):
    
    sentence_grouped = pos_tag(word_tokenize(sentence))
    
    return sentence_grouped    

def make_predictions(document):

	tagged_list = []
	final_list = []

	features_list = []
	sent_matrix = []

	sentences = sent_tokenize(document)

	for sentence in sentences:
		sent_matrix.append(word_tokenize(sentence))

	for sentence in sentences:
		sentence = preprocess(sentence)
		features = get_features(sentence)
		features_list.append(features)

	label_matrix = crf_model.predict(features_list)

	for i in range(len(label_matrix)):
		final_list.append(zip(sent_matrix[i], label_matrix[i]))

	for sent_ in final_list:
		for word_ in sent_:
			tagged_list.append(word_)

	return tagged_list

#print("Loading model...")
#get_model()

@app.route("/", methods=['GET','POST'])
def index():
	if request.method == 'GET':
		return render_template('page.html')
	else:

		input_text = request.get_json(force=True)
		
		doc = input_text['text']
		
		get_model()

		# *********************************************
		# logic
		tagged = make_predictions(doc)
		# *********************************************

		#tagged = [('I','O'),('call','O')]

		json_list = []

		for word,tag in tagged:
			json_item = {
				"word" : word,
				"tag"  : tag
			}

			json_list.append(json_item)

		#print(json_list)

		response = {
			'json_list' : json_list
		}

		#print(jsonify(response)) 

		return jsonify(response)

if __name__ == '__main__':
	app.run(host='0.0.0.0')
