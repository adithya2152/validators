# import warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import logging
# logging.getLogger('absl').setLevel(logging.ERROR)
# from app.analytics.utils.helper import extract_faces_from_frames
# from app.analytics.utils.video import eyebrow,detect_blinks,detect_yawns,detect_smiles
# from app.analytics.utils.valence_arousal_detection import va_predict
# from app.analytics.utils.facial_emotion_recognition import fer_predict,plot_fer_graph
# from app.analytics.utils.helper import plot_facial_expression_graphs
# from app.analytics.utils.audio import extract_audio_features
# from app.analytics.utils.speech_util import AudioAnalysis
# from app.analytics.utils.facs import calculate_au_percentages, au_columns, predict_au_from_video
# from moviepy.editor import VideoFileClip
# import json 
# import pandas as pd 
# from typing import Callable
# import traceback
# from app.lib.api import API
# import shutil
# import time
# from pathlib import Path
# from app.config import API_CONFIG
# import pytz
# from datetime import datetime

# fps=30
# session_data={}
# calibration={}



# def analyze_interview(models: dict, video_path: str, session_id: str, question_id: str, baseline_questions: list[str], video_count: int, is_last_video: bool, save: bool):
# 	api = API(session_id, video_count, question_id, save)
# 	try:
# 		api.log(f"Analyzing video for question")
# 		asrmodel=models['asrmodel']
# 		asrproc=models['asrproc']
# 		sentipipe=models['sentipipe']
# 		valence_arousal_model=models['valence_fer'][1]
# 		val_ar_feat_model=models['valence_fer'][0]
# 		fer_model=models['fer']
# 		smile_cascade=models['smile_cascade']
# 		dnn_net=models['face'][0]
# 		predictor=models['face'][1]
# 		# gem_model=models['gem']
# 		audio_analysis: AudioAnalysis =models['audio_analysis']
# 		facs_model=models['facs']
	
# 		global session_data
# 		global calibration

# 		if session_id not in session_data:
# 				session_data[session_id]={
# 						"vcount":[],
# 						"duration":[],      
# 						"audio":[],
# 						"pitches":[],
# 						"blinks":[],
# 						"yawn":[],
# 						"smile":[],
# 						"eyebrow":[],
# 						"fer": [],
# 						"valence":[],
# 						"arousal":[],
# 						"stress":[],
# 						"sentiment":[],
# 						"processed_videos":[],
# 						"facs": [],
# 						"au_percentage": [],
# 						"baseline": {}
# 				}


# 		SESSION_FOLDER = Path(f'{API_CONFIG["DATA_FOLDER"]}/{session_id}')
# 		OUTPUT_FOLDER = Path(f'{API_CONFIG["DATA_FOLDER"]}/{session_id}/outputs/{video_count}')
# 		OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
			

		


# 		meta_data_path=os.path.join(OUTPUT_FOLDER,'metadata.json')
# 		valence_plot=os.path.join(OUTPUT_FOLDER,"vas.png")
# 		ratio_plot=os.path.join(OUTPUT_FOLDER,"ratio.png")
# 		word_cloud=os.path.join(OUTPUT_FOLDER,'wordcloud.jpg')
# 		audio_metrics_comparison_plot = os.path.join(OUTPUT_FOLDER, 'audio_metrics_comparison.png')
# 		valence_combined_plot=os.path.join(OUTPUT_FOLDER,"vas_combined.png")
# 		ratio_combined_plot=os.path.join(OUTPUT_FOLDER,"ratio_combined.png")
# 		pdf_filename = os.path.join(OUTPUT_FOLDER,"formatted_output_with_plots.pdf")

# 		print(f"Loading video from: ", video_path)
# 		video_clip=VideoFileClip(video_path)
# 		video_clip=video_clip.set_fps(fps)
# 		duration=video_clip.duration
# 		session_data[session_id]['vcount'].append(video_count)
# 		session_data[session_id]['duration'].append(duration)
# 		audio=video_clip.audio
# 		audio_path = os.path.join(OUTPUT_FOLDER,'extracted_audio.wav')
# 		audio.write_audiofile(audio_path)
# 		video_frames=[frame for frame in video_clip.iter_frames()]
# 		faces, landmarks, sizes=extract_faces_from_frames(video_frames,dnn_net,predictor)

# 		# faces=[extract_face(frame) for frame in tqdm(video_frames)]
# 		af,pitches=extract_audio_features(audio_path,asrmodel,asrproc,sentipipe,duration,word_cloud)
# 		pitches=[float(pitch) for pitch in pitches]

# 		api.log(f"Extracting facial emotion features for question")
# 		fer_emotions,class_wise_frame_count,em_tensors=fer_predict(faces,fps,fer_model)
# 		valence_list,arousal_list=va_predict(valence_arousal_model,val_ar_feat_model,faces,list(em_tensors))

# 		#FACS
# 		api.log(f"Extracting facial action coding system features for question")
# 		au_predictions=predict_au_from_video(facs_model,faces)
# 		au_percentages=calculate_au_percentages(au_predictions)
# 		session_data[session_id]['au_percentage'].append(au_percentages)
# 		session_data[session_id]['facs'].append(au_predictions)
# 		au_df = pd.DataFrame(au_predictions, columns=au_columns)
# 		au_path = os.path.join(OUTPUT_FOLDER,'au_predictions.csv')
# 		au_df.to_csv(au_path, index=False)

# 		timestamps=[j/fps for j in range(len(valence_list))]

# 		api.log(f"Extracting eye features for question")
# 		eyebrow_dist=eyebrow(landmarks,sizes)

# 		blinks,blink_count, ear_ratios=detect_blinks(landmarks,sizes,fps)
# 		ear_ratios=[float(pitch) for pitch in ear_ratios]

# 		smiles, smile_ratios, total_smiles, smile_durations,smile_threshold=detect_smiles(landmarks,sizes)
# 		smile_ratios=[float(smile) for smile in smile_ratios]

# 		yawns, yawn_ratios, total_yawns, yawn_durations=detect_yawns(landmarks,sizes)

# 		thresholds=[smile_threshold,0.225,0.22]
		
# 		plot_facial_expression_graphs(smile_ratios, ear_ratios, yawn_ratios, thresholds, full_path=ratio_plot)

# 		y_vals = [valence_list, arousal_list,eyebrow_dist,pitches]
# 		labels = ['Valence', 'Arousal',"EyeBrowDistance","Pitch"]
		
# 		plot_fer_graph(timestamps, y_vals, labels, full_path=valence_plot)

# 		# Voice module
# 		output_audio_path = os.path.join(OUTPUT_FOLDER, "output_audio.mp3")
# 		audio_analysis.extractAudio(output_audio_path, video_path)
# 		audio_metrics = audio_analysis.analyzeAudio(output_audio_path)
# 		session_data[session_id]['baseline'][question_id] = audio_metrics

# 		meta_data={}
# 		meta_data['duration']=duration
# 		meta_data['facs'] = {
# 			"AU_percentages":au_percentages,
# 		}
# 		meta_data['facial_emotion_recognition'] = {
# 			"class_wise_frame_count": class_wise_frame_count,
# 		}
# 		meta_data['audio']=af

# 		if len(baseline_questions) > 0:
# 			if not all(baseline_question in list(session_data[session_id]['baseline'].keys()) for baseline_question in baseline_questions):
# 				api.log(f"Waiting - Processed: {len(session_data[session_id]['baseline'].keys())}, Baseline questions: {len(baseline_questions)}")
# 			while not all(baseline_question in list(session_data[session_id]['baseline'].keys()) for baseline_question in baseline_questions):
# 				time.sleep(3)
# 			baseline_metrics_list = []
# 			for baseline_question in baseline_questions:
# 				if baseline_question in session_data[session_id]['baseline']:
# 					baseline_metrics_list.append(session_data[session_id]['baseline'][baseline_question])
# 				else:
# 					raise ValueError(f"Baseline question '{baseline_question}' not found in session data.")
					
# 			audio_analysis_result = audio_analysis.compareAgainstBaseline(audio_metrics_comparison_plot, baseline_metrics_list, audio_metrics)	
			
# 			meta_data['audio_analysis'] = audio_analysis_result

# 		with open(meta_data_path, 'w') as json_file:
# 				json.dump(meta_data, json_file, indent=4)

# 		session_data[session_id]['audio'].append(af)
# 		session_data[session_id]['pitches'].append(pitches)
# 		session_data[session_id]['fer'].append(fer_emotions)
# 		session_data[session_id]['valence'].append(valence_list)
# 		session_data[session_id]['arousal'].append(arousal_list)
# 		session_data[session_id]['eyebrow'].append(eyebrow_dist)
# 		session_data[session_id]['smile'].append(smile_ratios)
# 		session_data[session_id]['blinks'].append(ear_ratios)
# 		session_data[session_id]['yawn'].append(yawn_ratios)
# 		session_data[session_id]['sentiment'].append(af['sentiment'][0]['label'])
# 		if video_count==1:
# 				try:	
# 						filtered_val=[item for item in valence_list if isinstance(item, (int,float))]
# 						filtered_aro=[item for item in arousal_list if isinstance(item, (int,float))]
# 						calibration_valence=sum(filtered_val)/len(valence_list)
# 						calibration_arousal=sum(filtered_aro)/len(arousal_list)
# 						calibration_pitch=sum(pitches)/len(pitches)
# 						calibration_eyebrow=sum(eyebrow_dist)/len(eyebrow_dist)
# 				except:
# 						calibration_arousal=0
# 						calibration_valence=0
# 						calibration_pitch=0
# 						calibration_eyebrow=0
# 				calibration['valence']=calibration_valence
# 				calibration['arousal']=calibration_arousal
# 				calibration['pitch']=calibration_pitch
# 				calibration['eyebrow']=calibration_eyebrow
		
# 		api.log(f"✅ Individual metadata has been generated")
# 		api.save_metadata(meta_data)
# 		api.save_file(f"{session_id}-{video_count}-vas.png", valence_plot)
# 		api.save_file(f"{session_id}-{video_count}-ratio.png", ratio_plot)
# 		api.save_file(f"{session_id}-{video_count}-wordcloud.jpg", word_cloud)
# 		api.save_file(f'{session_id}-{video_count}-au.csv', au_path)
# 		if os.path.exists(audio_metrics_comparison_plot):
# 			api.save_file(f"{session_id}-{video_count}-audio_metrics_comparison.png", audio_metrics_comparison_plot)
# 		session_data[session_id]['processed_videos'].append(True)

# 		api.log(f"Processed Video")
		
# 		if not is_last_video:
# 			return

# 		api.log(f"Waiting for videos, processed: {len(session_data[session_id]['processed_videos'])}")

# 		# Wait for all videos to be processed
# 		while len(session_data[session_id]['processed_videos'])!=video_count:
# 			time.sleep(1) 

# 		api.log(f"Processing gathered data for final output")
		
# 		videos=len(session_data[session_id]['vcount'])
# 		final_score=0
# 		#combined calculation 
# 		combined_pdf=os.path.join(OUTPUT_FOLDER,'combined.pdf')
# 		transcripts=''
# 		combined_valence=[]
# 		combined_arousal=[]
# 		combined_au=[]
# 		combined_fer=[]
# 		combined_pitch=[]
# 		combined_eyebrow=[]
# 		combined_blinks=[]
# 		combined_yawn=[]
# 		senti_list=[]
# 		combined_smiles=[]
# 		vid_index=[]

# 		for i in range(videos):
# 				timestamps=[j/fps for j in range(len(session_data[session_id]['valence'][i]))]	
# 				for j in range(len(timestamps)):
# 						vid_index.append(i+1)
# 				transcripts+=session_data[session_id]['audio'][i]['transcript']
# 				combined_pitch+=session_data[session_id]['pitches'][i]
# 				combined_arousal+=session_data[session_id]['arousal'][i]
# 				combined_valence+=session_data[session_id]['valence'][i]
# 				for j in range(len(session_data[session_id]['facs'][i])):
# 					combined_au.append(session_data[session_id]['facs'][i][j])
# 				combined_fer+=session_data[session_id]['fer'][i]
# 				combined_blinks+=session_data[session_id]['blinks'][i]
# 				combined_eyebrow+=session_data[session_id]['eyebrow'][i]
# 				combined_smiles+=session_data[session_id]['smile'][i]
# 				combined_yawn+=session_data[session_id]['yawn'][i]
# 				senti_list.append(session_data[session_id]['sentiment'][i])

# 		sentiment_scores = {"Positive": 1, "Negative": -1, "Neutral": 0}
# 		total_score = sum(sentiment_scores[sentiment] for sentiment in senti_list)
# 		normalized_senti_score = total_score / len(senti_list)
# 		neg_val=sum([1 for val in combined_valence if isinstance(val, (int, float)) and val<calibration['valence']])/len(combined_valence)
# 		neg_ar=sum([1 for val in combined_arousal if isinstance(val, (int, float)) and val>calibration['arousal']])/len(combined_arousal)
# 		neg_ya=sum([1 for val in combined_yawn if val>0.225])/len(combined_yawn)
# 		neg_sm=sum([1 for val in combined_smiles if val<smile_threshold])/len(combined_smiles)
# 		avg_sentiment=(neg_ar+neg_val+neg_ya+neg_sm+normalized_senti_score)/5
# 		y_vals = [combined_valence, combined_arousal,combined_eyebrow,combined_pitch]
# 		labels = ['Valence', 'Arousal',"EyeBrowDistance","Pitch"]
# 		plot_fer_graph(timestamps, y_vals, labels, full_path=valence_combined_plot)
# 		thresholds=[smile_threshold,0.225,0.22]
# 		plot_facial_expression_graphs(combined_smiles, combined_blinks, combined_yawn, thresholds, full_path=ratio_combined_plot)

# 		timestamps=[i/fps for i in range(len(combined_arousal))]
# 		# Ensure all columns have the same length by taking the minimum length
# 		l = min(len(combined_fer), len(combined_valence), len(combined_arousal), len(combined_eyebrow), len(combined_blinks), len(combined_yawn), len(combined_smiles), len(combined_pitch), len(timestamps), len(vid_index))

# 		df = pd.DataFrame({
# 			'timestamps':timestamps[:l],
# 			'video_index': vid_index[:l],  # Add a column for video index
# 			'fer': combined_fer[:l],
# 			'valence': combined_valence[:l],
# 			'arousal': combined_arousal[:l],
# 			'eyebrow':combined_eyebrow[:l],
# 			'blinks':combined_blinks[:l],
# 			'yawn':combined_yawn[:l],
# 			'smiles':combined_smiles[:l],
# 			'pitches':combined_pitch[:l]
# 		})

# 		combined_au_percentages = calculate_au_percentages(combined_au)
# 		au_df = pd.DataFrame(combined_au, columns=au_columns)
# 		df = pd.concat([df, au_df], axis=1)
# 		df.to_csv(os.path.join(OUTPUT_FOLDER,'combined_data.csv'), index=False)

# 		api.save_metadata({
# 			"avg_sentiment": 1 - avg_sentiment,
# 			"facs": {
# 				'AU_percentages':combined_au_percentages,
# 			}
# 		}, True)
# 		api.save_file(f"{session_id}-combined-vas.png", valence_combined_plot)
# 		api.save_file(f"{session_id}-combined-ratio.png", ratio_combined_plot)
# 		api.save_file(f"{session_id}-combined-data.csv", os.path.join(OUTPUT_FOLDER,'combined_data.csv'))
# 		api.log(f"✅ Final metadata is generated")

# 		api.close_session()
# 		del session_data[session_id]
# 		shutil.rmtree(SESSION_FOLDER)

# 	except Exception as e:
# 			error_trace = traceback.format_exc()
# 			print(error_trace)
# 			api.save_error(str(e), error_trace)
# 			if is_last_video:
# 				del session_data[session_id]
# 				shutil.rmtree(SESSION_FOLDER)
						
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from app.analytics.utils.helper import extract_faces_from_frames
from app.analytics.utils.video import eyebrow,detect_blinks,detect_yawns,detect_smiles
from app.analytics.utils.valence_arousal_detection import va_predict
from app.analytics.utils.facial_emotion_recognition import fer_predict,plot_fer_graph
from app.analytics.utils.helper import plot_facial_expression_graphs
from app.analytics.utils.audio import extract_audio_features
from app.analytics.utils.speech_util import AudioAnalysis
from app.analytics.utils.facs import calculate_au_percentages, au_columns, predict_au_from_video
from moviepy.editor import VideoFileClip
from app.lib.validation import AnalyticsValidator
import json 
import pandas as pd 
from typing import Callable
import traceback
from app.lib.api import API
import shutil
import time
from pathlib import Path
from app.config import API_CONFIG
import pytz
from datetime import datetime
import logging

output_logger = logging.getLogger('function_outputs')
output_logger.setLevel(logging.INFO)
output_logger.propagate = False
if not output_logger.hasHandlers():
    output_handler = logging.FileHandler('outputs.log')
    output_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    output_handler.setFormatter(output_formatter)
    output_logger.addHandler(output_handler)

fps=30
session_data={}
calibration={}

def analyze_interview(models: dict, video_path: str, session_id: str, question_id: str, baseline_questions: list[str], video_count: int, is_last_video: bool, save: bool):
    api = API(session_id, video_count, question_id, save)
    validator = AnalyticsValidator()
    try:
        api.log(f"Analyzing video for question {question_id}")
        asrmodel, asrproc, sentipipe = models['asrmodel'], models['asrproc'], models['sentipipe']
        valence_arousal_model, val_ar_feat_model = models['valence_fer'][1], models['valence_fer'][0]
        fer_model, smile_cascade = models['fer'], models['smile_cascade']
        dnn_net, predictor = models['face'][0], models['face'][1]
        audio_analysis: AudioAnalysis = models['audio_analysis']
        facs_model = models['facs']
        
        global session_data
        global calibration

        if session_id not in session_data:
                session_data[session_id]={ "vcount":[], "duration":[], "audio":[], "pitches":[], "blinks":[], "yawn":[], "smile":[], "eyebrow":[], "fer": [], "valence":[], "arousal":[], "stress":[], "sentiment":[], "processed_videos":[], "facs": [], "au_percentage": [], "baseline": {} }

        SESSION_FOLDER = Path(f'{API_CONFIG["DATA_FOLDER"]}/{session_id}')
        OUTPUT_FOLDER = Path(f'{API_CONFIG["DATA_FOLDER"]}/{session_id}/outputs/{video_count}')
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        meta_data_path=os.path.join(OUTPUT_FOLDER,'metadata.json')
        valence_plot=os.path.join(OUTPUT_FOLDER,"vas.png")
        ratio_plot=os.path.join(OUTPUT_FOLDER,"ratio.png")
        word_cloud=os.path.join(OUTPUT_FOLDER,'wordcloud.jpg')
        audio_metrics_comparison_plot = os.path.join(OUTPUT_FOLDER, 'audio_metrics_comparison.png')
        valence_combined_plot=os.path.join(OUTPUT_FOLDER,"vas_combined.png")
        ratio_combined_plot=os.path.join(OUTPUT_FOLDER,"ratio_combined.png")

        api.log("Loading video and extracting frames/audio.")
        video_clip=VideoFileClip(video_path).set_fps(fps)
        duration=video_clip.duration
        session_data[session_id]['vcount'].append(video_count)
        session_data[session_id]['duration'].append(duration)
        audio=video_clip.audio
        audio_path = os.path.join(OUTPUT_FOLDER,'extracted_audio.wav')
        audio.write_audiofile(audio_path, logger=None)
        video_frames=[frame for frame in video_clip.iter_frames()]
        output_logger.info(f"Extracted {len(video_frames)} frames from video.")

        if not validator.validate_inputs('Face Extraction', api, video_frames, dnn_net, predictor):
            raise ValueError("Invalid inputs for Face Extraction module.")
        faces, landmarks, sizes = extract_faces_from_frames(video_frames, dnn_net, predictor)

        api.log("Extracting audio features.")
        if not validator.validate_inputs('Audio Features', api, audio_path, asrmodel, asrproc, sentipipe):
            raise ValueError("Invalid inputs for Audio Features module.")
        af, pitches = extract_audio_features(audio_path, asrmodel, asrproc, sentipipe, duration, word_cloud)
        pitches = [float(pitch) for pitch in pitches]

        api.log("Extracting facial emotion features for question.")
        if not validator.validate_inputs('FER', api, faces, fps, fer_model):
            raise ValueError("Invalid inputs for FER module.")
        fer_emotions, class_wise_frame_count, em_tensors = fer_predict(faces, fps, fer_model)

        api.log("Extracting valence and arousal features.")
        if not validator.validate_inputs('Valence Arousal', api, valence_arousal_model, val_ar_feat_model, faces, list(em_tensors)):
            raise ValueError("Invalid inputs for Valence Arousal module.")
        valence_list, arousal_list = va_predict(valence_arousal_model, val_ar_feat_model, faces, list(em_tensors))

        api.log("Extracting facial action coding system features for question.")
        if not validator.validate_inputs('FACS', api, facs_model, faces):
            raise ValueError("Invalid inputs for FACS module.")
        au_predictions = predict_au_from_video(facs_model, faces)
        au_percentages = calculate_au_percentages(au_predictions)

        api.log("Extracting eye, smile, and yawn features.")
        if not validator.validate_inputs('Behavioral Features', api, landmarks, sizes):
            raise ValueError("Invalid inputs for Behavioral Features.")
        eyebrow_dist = eyebrow(landmarks, sizes)
        blinks, blink_count, ear_ratios = detect_blinks(landmarks, sizes, fps)
        smiles, smile_ratios, total_smiles, smile_durations, smile_threshold = detect_smiles(landmarks, sizes)
        yawns, yawn_ratios, total_yawns, yawn_durations = detect_yawns(landmarks, sizes)
        ear_ratios = [float(pitch) for pitch in ear_ratios]
        smile_ratios = [float(smile) for smile in smile_ratios]
        
        output_audio_path = os.path.join(OUTPUT_FOLDER, "output_audio.mp3")
        audio_analysis.extractAudio(output_audio_path, video_path)
        if not validator.validate_inputs('Audio Analysis', api, output_audio_path):
            raise ValueError("Invalid inputs for Audio Analysis module.")
        audio_metrics = audio_analysis.analyzeAudio(output_audio_path)
        
        session_data[session_id]['au_percentage'].append(au_percentages)
        session_data[session_id]['facs'].append(au_predictions)
        au_df = pd.DataFrame(au_predictions, columns=au_columns)
        au_path = os.path.join(OUTPUT_FOLDER,'au_predictions.csv')
        au_df.to_csv(au_path, index=False)

        plot_facial_expression_graphs(smile_ratios, ear_ratios, yawn_ratios, [smile_threshold,0.225,0.22], full_path=ratio_plot)
        plot_fer_graph([j/fps for j in range(len(valence_list))], [valence_list, arousal_list,eyebrow_dist,pitches], ['Valence', 'Arousal',"EyeBrowDistance","Pitch"], full_path=valence_plot)

        session_data[session_id]['baseline'][question_id] = audio_metrics

        meta_data={'duration':duration, 'facs':{"AU_percentages":au_percentages}, 'facial_emotion_recognition':{"class_wise_frame_count": class_wise_frame_count}, 'audio':af}

        if len(baseline_questions) > 0:
            while not all(q in list(session_data[session_id]['baseline'].keys()) for q in baseline_questions):
                api.log(f"Waiting for baseline videos... Processed: {len(session_data[session_id]['baseline'].keys())}/{len(baseline_questions)}")
                time.sleep(3)
            baseline_metrics_list = [session_data[session_id]['baseline'][q] for q in baseline_questions if q in session_data[session_id]['baseline']]
            if len(baseline_metrics_list) != len(baseline_questions): raise ValueError("Mismatch in required vs available baseline questions.")
            audio_analysis_result = audio_analysis.compareAgainstBaseline(audio_metrics_comparison_plot, baseline_metrics_list, audio_metrics)  
            meta_data['audio_analysis'] = audio_analysis_result

        with open(meta_data_path, 'w') as json_file:
            json.dump(meta_data, json_file, indent=4)

        session_data[session_id]['audio'].append(af)
        session_data[session_id]['pitches'].append(pitches)
        session_data[session_id]['fer'].append(fer_emotions)
        session_data[session_id]['valence'].append(valence_list)
        session_data[session_id]['arousal'].append(arousal_list)
        session_data[session_id]['eyebrow'].append(eyebrow_dist)
        session_data[session_id]['smile'].append(smile_ratios)
        session_data[session_id]['blinks'].append(ear_ratios)
        session_data[session_id]['yawn'].append(yawn_ratios)
        session_data[session_id]['sentiment'].append(af['sentiment'][0]['label'])
        
        if video_count == 1:
            filtered_val = [v for v in valence_list if isinstance(v, (int, float))]
            filtered_aro = [a for a in arousal_list if isinstance(a, (int, float))]
            filtered_pitch = [p for p in pitches if isinstance(p, (int, float))]
            filtered_eyebrow = [e for e in eyebrow_dist if isinstance(e, (int, float))]
            calibration['valence'] = sum(filtered_val) / len(filtered_val) if filtered_val else 0
            calibration['arousal'] = sum(filtered_aro) / len(filtered_aro) if filtered_aro else 0
            calibration['pitch'] = sum(filtered_pitch) / len(filtered_pitch) if filtered_pitch else 0
            calibration['eyebrow'] = sum(filtered_eyebrow) / len(filtered_eyebrow) if filtered_eyebrow else 0
        
        api.log(f"✅ Individual metadata for video {video_count} has been generated.")
        api.save_metadata(meta_data)
        api.save_file(f"{session_id}-{video_count}-vas.png", valence_plot)
        api.save_file(f"{session_id}-{video_count}-ratio.png", ratio_plot)
        api.save_file(f"{session_id}-{video_count}-wordcloud.jpg", word_cloud)
        api.save_file(f'{session_id}-{video_count}-au.csv', au_path)
        if os.path.exists(audio_metrics_comparison_plot):
            api.save_file(f"{session_id}-{video_count}-audio_metrics_comparison.png", audio_metrics_comparison_plot)
        session_data[session_id]['processed_videos'].append(True)
        
        if not is_last_video:
            api.log("Finished single video analysis.")
            return

        api.log(f"Waiting for all videos, processed: {len(session_data[session_id]['processed_videos'])} of {video_count}")
        while len(session_data[session_id]['processed_videos']) != video_count:
            time.sleep(1) 

        api.log("Processing gathered data for final combined output.")
        videos = len(session_data[session_id]['vcount'])
        combined_valence, combined_arousal, combined_au, combined_fer, combined_pitch = [], [], [], [], []
        combined_eyebrow, combined_blinks, combined_yawn, senti_list, combined_smiles, vid_index = [], [], [], [], [], []

        for i in range(videos):
            vid_index.extend([i+1] * len(session_data[session_id]['valence'][i]))
            combined_pitch.extend(session_data[session_id]['pitches'][i])
            combined_arousal.extend(session_data[session_id]['arousal'][i])
            combined_valence.extend(session_data[session_id]['valence'][i])
            combined_au.extend(session_data[session_id]['facs'][i])
            combined_fer.extend(session_data[session_id]['fer'][i])
            combined_blinks.extend(session_data[session_id]['blinks'][i])
            combined_eyebrow.extend(session_data[session_id]['eyebrow'][i])
            combined_smiles.extend(session_data[session_id]['smile'][i])
            combined_yawn.extend(session_data[session_id]['yawn'][i])
            senti_list.append(session_data[session_id]['sentiment'][i])

        normalized_senti_score = sum({"Positive": 1, "Negative": -1, "Neutral": 0}[s] for s in senti_list) / len(senti_list)
        neg_val = sum(1 for v in combined_valence if isinstance(v, (int, float)) and v < calibration.get('valence', 0)) / (len(combined_valence) or 1)
        neg_ar = sum(1 for v in combined_arousal if isinstance(v, (int, float)) and v > calibration.get('arousal', 0)) / (len(combined_arousal) or 1)
        neg_ya = sum(1 for v in combined_yawn if v > 0.225) / (len(combined_yawn) or 1)
        neg_sm = sum(1 for v in combined_smiles if v < smile_threshold) / (len(combined_smiles) or 1)
        avg_sentiment = (neg_ar + neg_val + neg_ya + neg_sm + normalized_senti_score) / 5

        l = min(len(combined_fer), len(combined_valence), len(combined_arousal), len(combined_eyebrow), len(combined_blinks), len(combined_yawn), len(combined_smiles), len(combined_pitch), len(vid_index))
        df_data = { 'timestamps': [i/fps for i in range(l)], 'video_index': vid_index[:l], 'fer': combined_fer[:l], 'valence': combined_valence[:l], 'arousal': combined_arousal[:l], 'eyebrow':combined_eyebrow[:l], 'blinks':combined_blinks[:l], 'yawn':combined_yawn[:l], 'smiles':combined_smiles[:l], 'pitches':combined_pitch[:l] }
        df = pd.DataFrame(df_data)
        au_df = pd.DataFrame(combined_au, columns=au_columns)
        df = pd.concat([df, au_df.iloc[:len(df)]], axis=1)
        df.to_csv(os.path.join(OUTPUT_FOLDER,'combined_data.csv'), index=False)

        api.save_metadata({"avg_sentiment": 1 - avg_sentiment, "facs": {'AU_percentages': calculate_au_percentages(combined_au)}}, True)
        api.save_file(f"{session_id}-combined-vas.png", valence_combined_plot)
        api.save_file(f"{session_id}-combined-ratio.png", ratio_combined_plot)
        api.save_file(f"{session_id}-combined-data.csv", os.path.join(OUTPUT_FOLDER,'combined_data.csv'))
        api.log("✅ Final combined metadata is generated.")
        
        api.close_session()
        del session_data[session_id]
        shutil.rmtree(SESSION_FOLDER)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        api.save_error(str(e), error_trace)
        api.log("Analysis failed.")
        if is_last_video and session_id in session_data:
            del session_data[session_id]
            shutil.rmtree(SESSION_FOLDER)