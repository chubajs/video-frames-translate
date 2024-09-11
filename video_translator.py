import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip
import tempfile
import base64
import json
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
from tqdm import tqdm
from datetime import datetime
import ast  # Add this import at the top of the file
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from scipy.stats import iqr

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Set up logging
log_filename = f"logs/video_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logger('video_translator', log_filename)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_video_hash(video_path):
    hasher = hashlib.md5()
    with open(video_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_or_create_index(video_path):
    video_hash = get_video_hash(video_path)
    index_path = f"tmp/{video_hash}_index.json"
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}

def save_index(index, video_path):
    video_hash = get_video_hash(video_path)
    index_path = f"tmp/{video_hash}_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_frame(frame):
    original_height, original_width = frame.shape[:2]
    logger.info(f"Original frame size: {original_width}x{original_height}")

    # Resize image to have longest side at 512 pixels
    if original_width > original_height:
        new_width = 512
        new_height = int(512 * original_height / original_width)
    else:
        new_height = 512
        new_width = int(512 * original_width / original_height)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    logger.info(f"Resized frame for API: {new_width}x{new_height}")

    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        pil_image.save(temp_file, format="JPEG")
        temp_filename = temp_file.name

    with open(temp_filename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = """
    Analyze this image and describe any text present. For each line of text, provide:
    1) its content
    2) the top-left corner coordinates (x1, y1) of the area containing the entire line
    3) the bottom-right corner coordinates (x2, y2) of the area containing the entire line

    If there's no text, just respond with: {"text_lines": []}

    Respond in the following JSON format:
    {
        "text_lines": [
            {
                "content": "Example text line 1",
                "top_left_corner": {
                    "x": 10,
                    "y": 20
                },
                "bottom_right_corner": {
                    "x": 210,
                    "y": 50
                }
            },
            {
                "content": "Example text line 2",
                "top_left_corner": {
                    "x": 15,
                    "y": 60
                },
                "bottom_right_corner": {
                    "x": 195,
                    "y": 85
                }
            }
        ]
    }
    """

    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                ],
            }
        ],
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    os.unlink(temp_filename)

    analysis = json.loads(response.choices[0].message.content)
    logger.info(f"GPT Analysis: {json.dumps(analysis, indent=2)}")
    
    text_blocks = []
    if 'text_lines' in analysis:
        text_blocks = process_lines(analysis['text_lines'])

    logger.info(f"Detected {len(text_blocks)} text blocks")
    return {
        'text_blocks': text_blocks,
        'resized_width': new_width,
        'resized_height': new_height,
        'original_width': original_width,
        'original_height': original_height
    }

def process_lines(lines):
    text_blocks = []
    for i, line in enumerate(lines):
        try:
            x1 = line['top_left_corner']['x']
            y1 = line['top_left_corner']['y']
            x2 = line['bottom_right_corner']['x']
            y2 = line['bottom_right_corner']['y']
            content = line['content']
            width = x2 - x1
            height = y2 - y1
            if all([x1, y1, x2, y2, content]) and width > 0 and height > 0:
                text_blocks.append({
                    'text': content,
                    'position': [x1, y1, width, height]
                })
                logger.info(f"Text block {i+1}: Content: '{content}', API position: [x={x1}, y={y1}, w={width}, h={height}]")
            else:
                logger.warning(f"Invalid data for text block {i+1}: {line}")
        except KeyError as e:
            logger.error(f"Missing key in text_line data: {e}")
    return text_blocks

def translate_text(text):
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the following text from English to Russian. Provide only the translation, without any additional comments or formatting."},
            {"role": "user", "content": text}
        ],
        max_tokens=300
    )
    translated_text = response.choices[0].message.content.strip()
    logger.info(f"Original text: {text}")
    logger.info(f"Translated text: {translated_text}")
    return translated_text

def get_optimal_font_scale(text, max_width, max_height, font_path, min_size=10, max_size=100):
    for font_size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= max_width and text_height <= max_height:
            return font_size
    return min_size

def create_translated_frame(frame, text_blocks, verbose=False):
    new_frame = frame.copy()
    frame_height, frame_width = new_frame.shape[:2]
    if verbose:
        logger.info(f"Frame size for translation: {frame_width}x{frame_height}")
    
    font_path = "fonts/Rubik-Medium.ttf"  # Ensure this path is correct
    
    # Find the smallest font size that fits all text blocks
    min_font_size = float('inf')
    for block in text_blocks:
        translated_text = block['translation']
        _, _, width, height = map(int, block['position'])
        font_size = get_optimal_font_scale(translated_text, width, height, font_path)
        min_font_size = min(min_font_size, font_size)
    
    font = ImageFont.truetype(font_path, min_font_size)
    
    # Group text blocks by unique translations
    unique_translations = {}
    for block in text_blocks:
        translation = block['translation']
        if translation not in unique_translations:
            unique_translations[translation] = block
        else:
            # Average the position if the translation already exists
            existing = unique_translations[translation]
            new_x = (existing['position'][0] + block['position'][0]) // 2
            new_y = (existing['position'][1] + block['position'][1]) // 2
            new_width = max(existing['position'][2], block['position'][2])
            new_height = max(existing['position'][3], block['position'][3])
            unique_translations[translation] = {
                'text': block['text'],
                'translation': translation,
                'position': [new_x, new_y, new_width, new_height]
            }
    
    for block in unique_translations.values():
        original_text = block['text']
        translated_text = block['translation']
        x, y, width, height = map(int, block['position'])
        
        if verbose:
            logger.info(f"Text block before drawing:")
            logger.info(f"  Original: {original_text}")
            logger.info(f"  Translated: {translated_text}")
            logger.info(f"  Position: x={x}, y={y}, w={width}, h={height}")
        
        # Apply Gaussian blur to the original text area
        roi = new_frame[y:y+height, x:x+width]
        blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
        new_frame[y:y+height, x:x+width] = blurred_roi
        
        # Fill original text area with semi-transparent black
        overlay = new_frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, new_frame, 0.4, 0, new_frame)
        
        # Convert to PIL Image for text drawing
        pil_image = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate text position within the original area
        text_bbox = draw.textbbox((0, 0), translated_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2
        text_y = y + (height - text_height) // 2
        
        # Draw translated text
        draw.text((text_x, text_y), translated_text, font=font, fill=(255, 255, 255))  # White text
        
        # Convert back to OpenCV format
        new_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if verbose:
            logger.info(f"Text block translation:")
            logger.info(f"  Original: {original_text}")
            logger.info(f"  Translated: {translated_text}")
            logger.info(f"  Original position: x={x}, y={y}, w={width}, h={height}")
            logger.info(f"  Translated text size: w={text_width}, h={text_height}")
            logger.info(f"  Translated text position: x={text_x}, y={text_y}")
    
    return new_frame

def get_frame_hash(frame):
    # Convert frame to grayscale and resize for faster comparison
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return resized.flatten()

def are_frames_similar(hash1, hash2, threshold=0.95):
    similarity = cosine_similarity(hash1.reshape(1, -1), hash2.reshape(1, -1))[0][0]
    return similarity > threshold

def group_similar_frames(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_groups = defaultdict(list)
    current_group_id = 0
    last_hash = None
    
    for frame_count in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        
        frame_hash = get_frame_hash(frame)
        
        if last_hash is None or not are_frames_similar(frame_hash, last_hash):
            current_group_id += 1
            last_hash = frame_hash
        
        frame_groups[current_group_id].append((frame_count, frame))
        
        if frame_count % 100 == 0:
            logger.info(f"Grouping frame {frame_count}/{total_frames}")
    
    video.release()
    return frame_groups, fps

def analyze_multiple_frames(frames, num_samples=5):
    total_frames = len(frames)
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    all_text_blocks = []
    resized_width = None
    resized_height = None
    for idx in sample_indices:
        _, frame = frames[idx]
        analysis = analyze_frame(frame)
        all_text_blocks.extend(analysis['text_blocks'])
        logger.info(f"Analyzed frame {idx+1}/{len(sample_indices)} in group. Detected {len(analysis['text_blocks'])} text blocks.")
        if resized_width is None:
            resized_width = analysis['resized_width']
            resized_height = analysis['resized_height']
    
    # Group text blocks by content
    grouped_blocks = defaultdict(list)
    for block in all_text_blocks:
        grouped_blocks[block['text']].append(block['position'])
    
    averaged_blocks = []
    for text, positions in grouped_blocks.items():
        x_values = [p[0] for p in positions]
        y_values = [p[1] for p in positions]
        w_values = [p[2] for p in positions]
        h_values = [p[3] for p in positions]
        
        # Calculate average position
        avg_x = int(np.mean(x_values))
        avg_y = int(np.mean(y_values))
        avg_width = int(np.mean(w_values))
        avg_height = int(np.mean(h_values))
        
        averaged_blocks.append({
            'text': text,
            'position': [avg_x, avg_y, avg_width, avg_height]
        })
        logger.info(f"Averaged text block: '{text}', Position: x={avg_x}, y={avg_y}, w={avg_width}, h={avg_height}")
    
    logger.info(f"Total averaged text blocks: {len(averaged_blocks)}")
    return {
        'text_blocks': averaged_blocks,
        'resized_width': resized_width,
        'resized_height': resized_height
    }

def process_video(input_path, output_path):
    logger.info(f"Starting video processing for {input_path}")
    
    frame_groups, fps = group_similar_frames(input_path)
    logger.info(f"Grouped frames into {len(frame_groups)} groups")
    
    original_video = cv2.VideoCapture(input_path)
    total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_video.release()
    
    logger.info(f"Original video dimensions: {original_width}x{original_height}")
    
    translated_groups = {}
    for group_id, frames in frame_groups.items():
        logger.info(f"Analyzing group {group_id} with {len(frames)} frames")
        
        # Analyze multiple frames
        analysis = analyze_multiple_frames(frames, num_samples=5)
        
        if analysis['text_blocks']:
            # Collect all text and their positions
            all_text = []
            all_positions = []
            for block in analysis['text_blocks']:
                all_text.append(block['text'])
                x, y, width, height = block['position']
                
                # Interpolate coordinates to original video dimensions
                x = int(x * original_width / analysis['resized_width'])
                y = int(y * original_height / analysis['resized_height'])
                width = int(width * original_width / analysis['resized_width'])
                height = int(height * original_height / analysis['resized_height'])
                
                all_positions.append([x, y, width, height])
            
            # Combine all text into a single string
            combined_text = " ".join(all_text)
            
            # Translate the combined text
            translated_text = translate_text(combined_text)
            
            # Ask ChatGPT to split the translated text
            split_text = split_translated_text(translated_text, len(all_text))
            
            # Create text blocks with translated text and original positions
            text_blocks = []
            for i, (text, position) in enumerate(zip(split_text, all_positions)):
                x, y, width, height = position
                
                # Increase area by 30% (15% on each side)
                width_increase = int(width * 0.3)
                height_increase = int(height * 0.3)
                
                # Calculate new dimensions
                new_width = min(width + width_increase, original_width)
                new_height = min(height + height_increase, original_height)
                
                # Adjust x and y to keep the text centered
                x_offset = width_increase // 2
                y_offset = height_increase // 2
                
                # Move the area up by 20% of the new height
                y_move_up = int(new_height * 0.2)
                
                # Adjust coordinates
                new_x = max(0, x - x_offset)
                new_y = max(0, y - y_offset - y_move_up)
                
                # Ensure we don't exceed frame borders
                new_x = min(new_x, original_width - new_width)
                new_y = min(new_y, original_height - new_height)
                
                text_blocks.append({
                    'text': all_text[i],
                    'translation': text,
                    'position': [new_x, new_y, new_width, new_height]
                })
            
            translated_groups[group_id] = text_blocks
            logger.info(f"Group {group_id}: Processed {len(text_blocks)} text blocks")
        else:
            logger.info(f"No text detected in group {group_id}")
    
    if not translated_groups:
        logger.info("No text detected in any frame group. Aborting video creation.")
        return
    
    logger.info("Creating output video...")
    original_video = cv2.VideoCapture(input_path)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_output_path = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        for group_id, frames in frame_groups.items():
            group_progress = tqdm(frames, desc=f"Group {group_id}", leave=False, unit="frame")
            
            if group_id in translated_groups:
                # Create a single translated frame for the group
                first_frame = frames[0][1]
                translated_frame = create_translated_frame(first_frame, translated_groups[group_id], verbose=True)
                logger.info(f"Created translated frame for group {group_id}")
            else:
                translated_frame = None
                logger.info(f"No translation for group {group_id}")
            
            for _, frame in group_progress:
                if translated_frame is not None:
                    out.write(translated_frame)
                else:
                    out.write(frame)
                frame_count += 1
                pbar.update(1)
            
            group_progress.close()
            logger.info(f"Processed group {group_id}, total frames: {frame_count}/{total_frames}")
    
    original_video.release()
    out.release()
    
    logger.info("Adding audio to the output video...")
    original_clip = VideoFileClip(input_path)
    new_video_clip = VideoFileClip(temp_output_path)
    final_video = new_video_clip.set_audio(original_clip.audio)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    new_video_clip.close()
    original_clip.close()
    os.remove(temp_output_path)
    
    logger.info("Video processing complete!")

def split_translated_text(translated_text, num_blocks):
    prompt = f"""
    Split the following translated text into {num_blocks} parts, maintaining the original meaning and structure as much as possible:

    {translated_text}

    Respond with a Python list of {num_blocks} strings, each representing a part of the split text. Do not include any additional formatting or code blocks.
    """
    
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that splits translated text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    content = response.choices[0].message.content.strip()
    
    # Try to extract a list from the content
    try:
        # First, try to evaluate as a Python literal
        split_text = ast.literal_eval(content)
    except (SyntaxError, ValueError):
        # If that fails, try to extract a list-like structure
        import re
        matches = re.findall(r'\[([^\]]+)\]', content)
        if matches:
            # Join all matches and try to evaluate again
            combined = '[' + ', '.join(matches) + ']'
            try:
                split_text = ast.literal_eval(combined)
            except (SyntaxError, ValueError):
                # If still fails, fall back to simple splitting
                split_text = content.split('\n')
        else:
            # If no list-like structure found, fall back to simple splitting
            split_text = content.split('\n')
    
    # Ensure we have the correct number of blocks
    if len(split_text) != num_blocks:
        logger.warning(f"Expected {num_blocks} blocks, but got {len(split_text)}. Adjusting...")
        if len(split_text) > num_blocks:
            split_text = split_text[:num_blocks]
        else:
            split_text.extend([''] * (num_blocks - len(split_text)))
    
    return split_text

if __name__ == "__main__":
    input_video = "tmp/sample1.mp4"
    output_video = "tmp/sample1_russian.mp4"
    process_video(input_video, output_video)