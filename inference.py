from pathlib import Path
from classification.src.classifiers import init_model
import torch
import utils as u
import numpy as np
from tqdm import tqdm
from preprocessing.src import feature_extractors
import skvideo.io
import pandas as pd
from preprocessing.src import video_utils as u_video
import argparse

extractor = feature_extractors.BEATSRunner(predict=True)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_dir', type=str, help='Path to the model directory',
                    default='classification/output/single_transformer')
parser.add_argument('--video_path', type=str, help='Path to the video file',
                    default=None,)
parser.add_argument('--youtube_link', type=str, required=False, help='Path to the video file',
                    default=None,)

args = parser.parse_args()
print("Parsing done")
assert bool(args.video_path) != bool(args.youtube_link), 'Provide either video path or YouTube link, and not both.'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if bool(args.video_path):
    video_path = Path(args.video_path)
else:
    target_dir = Path('preprocessing/data/youtube')
    target_dir.mkdir(exist_ok=True)
    video_path = Path(u_video.download_youtube(args.youtube_link, target_dir=target_dir))

model_dir = Path(args.model_dir)
groundtruth = pd.read_csv('preprocessing/data/labels/trailers_genres_clean.csv')

video_name = video_path.stem
if video_name in groundtruth['youtube_id'].values:
    groundtruth = eval(groundtruth.loc[groundtruth['youtube_id'] == video_name, 'genres'].values.item())
else:
    groundtruth = None

visualize = True
feed_tensor = True

use_scenecuts = True
fps = 1

labels = sorted(["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"])

config = torch.load(model_dir / 'config.pt')
config['device'] = device
feature_names = config['features']
stats = torch.load('preprocessing/data/stats.pt')

if use_scenecuts:
    input_frames, _ = u_video.video_to_midscenes(video_path)
else:
    input_frames = skvideo.io.vread(str(video_path))
    if fps is not None:
        input_fps = u_video.get_video_fps(str(video_path))
        multiplier = input_fps / fps
        indices = np.round(np.arange(0, input_frames.shape[0] - 2, multiplier)).astype(np.int32).tolist()
        input_frames = input_frames[indices, ...]

input_audio, sr = u_video.extract_audio(video_path)

sample_features = {}
video_outputs = {}
clip_features = None

with torch.no_grad():
    for feature_name in tqdm(feature_names, desc='Extracting features'):
        print(f'\nFeature: {feature_name}')

        if feature_name == 'clip':
            extractor = feature_extractors.CLIPRunner()
            input_tensor = input_frames
        elif feature_name == 'beats':
            extractor = feature_extractors.BEATSRunner(predict=True)
            input_tensor = input_audio
        elif feature_name == 'asr_sentiment':
            extractor = feature_extractors.ASRSentiment()
            input_tensor = input_audio
        elif feature_name == 'ocr_sentiment':
            extractor = feature_extractors.OCRPipeline()
            input_tensor = input_frames
        elif feature_name == 'face_emotion':
            extractor = feature_extractors.FaceExtractAndClassify()
            input_tensor = input_frames

        extractor.to_device(device)

        if feed_tensor:
            video_output = extractor.process_video(input_tensor=input_tensor, sr=sr)
        else:
            video_output = extractor.process_video(video_path=video_path, n_frames=None, fps=1)

        if feature_name == 'asr_sentiment':
            sentiment_feat = video_output.get('features', None)
            graph_feat = video_output.get('graph_features', None)

            if sentiment_feat not in (None, []):
                sentiment_feat = sentiment_feat if isinstance(sentiment_feat, torch.Tensor) else torch.tensor(sentiment_feat)
                if config['normalize']:
                    sentiment_feat = u.normalize(sentiment_feat, stats['asr_sentiment']['min'], stats['asr_sentiment']['max'])
                if config['standardize']:
                    sentiment_feat = u.standardize(sentiment_feat, stats['asr_sentiment']['mean'], stats['asr_sentiment']['std'])
                sentiment_feat = sentiment_feat.unsqueeze(0).to(device)
                sample_features['asr_sentiment'] = sentiment_feat

            if graph_feat not in (None, []):
                graph_feat = graph_feat if isinstance(graph_feat, torch.Tensor) else torch.tensor(graph_feat)
                if config['normalize']:
                    graph_feat = u.normalize(graph_feat, stats['asr_graph']['min'], stats['asr_graph']['max'])
                if config['standardize']:
                    graph_feat = u.standardize(graph_feat, stats['asr_graph']['mean'], stats['asr_graph']['std'])
                graph_feat = graph_feat.unsqueeze(0).to(device)
                sample_features['asr_graph'] = graph_feat

            video_outputs['asr_sentiment'] = video_output
            continue

        extracted_feature = video_output['features']
        if feature_name == 'clip':
            clip_features = extracted_feature

        video_outputs[feature_name] = video_output

        if extracted_feature == []:
            extracted_feature = torch.zeros((config['feature_lengths'][feature_name], config['feature_dims'][feature_name]))
        else:
            if config['normalize']:
                extracted_feature = u.normalize(extracted_feature, stats[feature_name]['min'], stats[feature_name]['max'])
            if config['standardize']:
                extracted_feature = u.standardize(extracted_feature, stats[feature_name]['mean'], stats[feature_name]['std'])

        source_length = extracted_feature.shape[0]
        target_length = config['feature_lengths'][feature_name]

        if source_length > target_length:
            inds = u.equidistant_indices(source_length, target_length)
            extracted_feature = extracted_feature[inds, :]
        elif source_length < target_length:
            extracted_feature = torch.nn.functional.pad(extracted_feature, (0, 0, 0, target_length - source_length))

        extracted_feature = extracted_feature.unsqueeze(0).to(device)
        sample_features[feature_name] = extracted_feature

    if visualize:
        caption_model = feature_extractors.CaptionRunner()

    del extractor
    model = init_model(config)
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=lambda storage, loc: storage))
    model.eval()

    with torch.cuda.amp.autocast(enabled=config['amp']):
        output = model(sample_features).squeeze()
        output = torch.nn.functional.softmax(output)
        output = u.detach_tensor(output)

if visualize:
    if 'ocr_sentiment' in video_outputs:
        ocr_text = video_outputs['ocr_sentiment']['ocr_processed']
        ocr_boxes = video_outputs['ocr_sentiment']['coordinates']
        if ocr_text:
            ocr_text = '. '.join(ocr_text)
            sentiment_prediction = video_outputs['ocr_sentiment']['predictions'][0].capitalize()
            sentiment_percentage = round(video_outputs['ocr_sentiment']['predictions'][1] * 100)
            print('\nOptical character recognition (OCR):')
            print(ocr_text)
            print('Sentiment analysis:')
            print(f"{sentiment_percentage}%  {sentiment_prediction}")
    else:
        print("OCR sentiment isn't in model output.")
        ocr_boxes = [None]

    if 'asr_sentiment' in video_outputs:
        asr_text = video_outputs['asr_sentiment'].get('asr', '')
        asr_language = video_outputs['asr_sentiment'].get('language', None)
        predictions = video_outputs['asr_sentiment'].get('predictions', [])

        if asr_text:
            print('\nAutomatic speech recognition (ASR):')
            if asr_language not in ('english', None):
                print(f'(Translated from {asr_language.capitalize()}.)')
            print(asr_text)
            if predictions:
                sentiment_prediction = predictions[0].capitalize()
                sentiment_percentage = round(predictions[1] * 100)
                print('Sentiment analysis:')
                print(f"{sentiment_percentage}%  {sentiment_prediction}")
    else:
        print("ASR sentiment isn't in model output.")

    if 'beats' in video_outputs:
        predictions = video_outputs['beats']['predictions']
        if predictions:
            print('\nAudio classification:')
            for prediction in predictions:
                print(f'{round(prediction[1] * 100):2}%  {prediction[0]}')
    else:
        print("BEATS isn't in model output.")

    if 'face_emotion' in video_outputs:
        face_boxes = video_outputs['face_emotion']['coordinates']
        face_predictions = video_outputs['face_emotion']['predictions']
    else:
        face_boxes = [None]
        face_predictions = [None]
        print("Face emotion isn't in model output.")

    i = u.find_common_nonzero_argmax(ocr_boxes, face_boxes)

    ocr_boxes_frame = []
    if ocr_boxes != [None] and ocr_boxes[i] is not None:
        ocr_boxes_frame = [pred[0] for pred in ocr_boxes[i]]

    ocr_predictions_frame = [None] * len(ocr_boxes_frame)

    face_boxes_frame = []
    face_predictions_frame = []
    if face_boxes != [None] and face_boxes[i] is not None:
        face_boxes_frame = [u.convert_to_points(box) for box in face_boxes[i]]
        face_predictions_frame = face_predictions[i]

    face_predictions_frame = [f'{round(pred[1] * 100):2}%  {pred[0].upper()}' for pred in face_predictions_frame]

    boxes = face_boxes_frame + ocr_boxes_frame
    box_labels = face_predictions_frame + ocr_predictions_frame

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    title = None
    if clip_features is not None:
        clip_feature = clip_features[i:i+1, ...].to(device)
        with torch.no_grad():
            caption = caption_model(clip_feature)
            title = f'Predicted caption: {caption.capitalize()}'

    selected_frame = input_frames[i]
    u.draw_boxes_on_image(selected_frame, boxes, labels=box_labels, title=title)

sorted_indices = np.argsort(output)[::-1]

print('\nPrediction:')
for idx in sorted_indices[:5]:
    print(f"{round(output[idx] * 100, 1):5}%  {labels[idx]}")

if groundtruth is not None:
    print('\nGround-truth:\n' + '\n'.join(groundtruth))
