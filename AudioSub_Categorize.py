import os
import glob
import sys
import subprocess 
import csv
import sqlite3
import numpy as np
import pandas as pd
import srt
import re
import time
import json
import ffmpeg
import whisper
import whisperx
from whisperx import diarize
import openai
APIKEY = os.environ["OPENAI_API_KEY"]
openai.api_key = APIKEY
from pydub import AudioSegment
from pyannote.audio import Pipeline

#PREFERENCES
clean_setup =  False
force_new_embeddings=False  #True
maxspend=9#9  DEFAULT 9 WHEN NOT TESTING: # Tokens for Turbo per  response of new category generation
minimum_length = 45 # input("minimum quote length?") # DEFAULT TO 20

# Set the path of the input folder containing .m4a files
  # Prompt the user for a list of categories
print("This script will auto-transcribe full sentences of >40 seconds, and then categorize each section of subtitle by the most similar embeddings to the categories provided. ")
print("\n \n Input a file directory, and the program will convert each .m4a file into a .wav, for OpenAI Whisper to automatically transcribe locally. The WhisperX Module will align timecode for each word as an .srt file")
print("the minimum duration of each section is set to combine subs to at least 60 seconds and then end on an end-of-sentence break marker such as .!?,;:--.")
print("OpenAI ADA model generates embeddings (a vector of associated and related meanings) for each subtitle, as well as embeddings for each category/research code provided.")
print("We use cosign similarity comparison and assume the most similar category to be the most relevant tag for each section of text \n \n")
input_file = input("To begin, provide the input file:  ")
input_string = "" # + input("Provide a list of categories (or research codes separated by commas)")
new_file = str("AIcoded_"+os.path.splitext(os.path.basename(input_file))[0] + ".txt")
#if input("Force new embeddings? y/n") == "y":
#    force_new_embeddings=True



def m4a_to_srt_categorized(input_file, input_string, output_file, db_file):

  db_file = "data.db"
  if clean_setup: #CLEAN THE DATABASE
    with open(db_file, 'w+') as f:
        f.write("")
    f.close()
  db_setup(db_file)

  if (".m4a" in input_file):
    srt_file = create_srt_transcript(setup_audiom4a(input_file), str(os.path.splitext(os.path.basename(input_file))[0] + ".srt"))
  elif ".srt" in input_file:
    srt_file = input_file
 #not in database    new_setup = True
  if len(fetch_embedded_subs(srt_file, db_file))<1 or clean_setup or force_new_embeddings: # if not in DB, setup
    print("SETTING UP NEW EMBEDDINGS, Load subs from .srt")
    load_subtitles(srt_file, db_file)
    get_embeddings(srt_file, db_file)
    newcodes=find_new_codes(srt_file)
    new_categories=get_categories(str(input_string+",\n"+str(newcodes)), db_file)
    output_transcript(srt_file)
    categorize_codes(srt_file, new_categories, new_file, db_file)
    revise_categories(srt_file, new_file, db_file)
  else: #try the database
    print("Setting up from DB, good boy")
    #db_setup(db_file)
    subtitles = fetch_embedded_subs(srt_file, db_file)
    output_transcript(srt_file)
    categories = fetch_embedded_categories(db_file)
    categorize_codes(srt_file,categories,new_file,db_file)
    revise_categories(srt_file, new_file, db_file)

    

"""
This script performs audio transcription and subtitles creation using OpenAI's Whisper model and SRT format. 

Usage:
    1. Set the path of the input folder containing .m4a files.
    2. Provide a list of categories (or research codes separated by commas).
    3. Run the script.

Requirements:
    1. Python 3.6 or higher
    2. OpenAI's Whisper model
    3. SQLite3

Functions:
    db_setup(db_file):
        This function creates a table in the SQLite database to store the subtitle data.

    write_subs_tosrt(newsubs, srt_file):
        This function writes the subtitles to a file in the SRT format.

    load_subtitles(srt_file, db_file):
        This function parses the SRT file and stores the subtitles in the SQLite database.

    setup(input_folder):
        This function prepares the input files by converting the .m4a files to .wav format and transcribing them to SRT format.

    create_srt_transcript(input_file, output_file, device):
        This function performs audio transcription using OpenAI's Whisper model and creates an SRT file.

Input:
    db_file (str): The path of the SQLite database file to store the subtitle data.
    input_file (str): The path of the input file .m4a
    input_string (str): A list of categories (or research codes separated by commas).
    minimum_length (int): The minimum length of a subtitle in seconds.

Output:
    Subtitle data is stored in the SQLite database and then categorized SRT files are generated for each input .m4a file.

"""
db_file = "data.db"
filename = os.path.splitext(os.path.basename(input_file))[0]
wav_output = filename + ".wav"

  
def db_setup(db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create the table if it doesn't exist
    c.execute("CREATE TABLE IF NOT EXISTS categories (category TEXT, category_embedding TEXT, occurrences REAL, avgSimilarity REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, srt_file TEXT, embeddings TEXT, category TEXT, similarity REAL)")
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()
    
def write_subs_tosrt(newsubs, srt_file):#="temp.srt"):
    #clear files to write project outputs
    with open(srt_file, 'w+') as f:
        f.write(srt.make_legal_content(srt.compose(newsubs)))
    f.close()
    print("written to file")
    return srt_file

def setup_audiom4a(input_file):
    #if "m4a" in input_file:
      filename = os.path.splitext(os.path.basename(input_file))[0]
      wavsound = filename + ".wav"
      audio = AudioSegment.from_file(input_file, format="m4a")
        # Export the AudioSegment object as a .wav file
      audio.export(wavsound, format="wav")
      print(input_file)
      print(wavsound)
      return (wavsound)
    
def load_subtitles(srt_file, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Open the subtitle file
    f = open(srt_file,"r")
    srt_text = f.read()
    f.close()
    subtitles = srt.parse(srt_text)
    # Parse the subtitles into list format
    # combine subs does not work???
    for sub in subtitles:
        # Get the start and end times, and convert them to seconds from the start of the file
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        # Get the text of the subtitle
        text = sub.content
        # Insert into the database
        c.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?,?)", (start, end, text, srt_file,  "None", "None", "None")) #embedding, category, similarity
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()
    
def get_embeddings(srt_file, db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the subtitles from the database
    c.execute("SELECT start,end,text FROM subtitles WHERE srt_file = ? and embeddings='None'", (srt_file,))
    subtitles = c.fetchall()
    # Get the text of the subtitles
    for time_start,time_end,sub_text in subtitles:
        print(".")
        time.sleep(60/450) #(delay_in_seconds) 60/ Rate limit per minute 
        response = openai.Embedding.create(model="text-embedding-ada-002", input=sub_text)
        embedding = response["data"][0]["embedding"]
        c.execute("UPDATE subtitles SET embeddings = ? WHERE srt_file = ? AND start = ? AND end = ? AND text = ? ", (json.dumps(embedding), srt_file, time_start, time_end,sub_text))
        # Commit the changes
        conn.commit()
    # Close the connection
    conn.close()
    print("Embedded meanings and associations found for every sub.")
    print(" ...")

def create_srt_transcript(input_file, output_file: str, device: str = "cuda") -> None:
    """
    Create an srt transcript from an audio file using the Whisper speech recognition model.

    Args:
    - input_file (str): The path to the input audio file.
    - output_file (str): The path to the output srt file.
    - device (str): The device to use for processing, either "cuda" or "cpu" (default "cuda").

    Returns:
    - None.

    Raises:
    - IOError: If the input file is not found or cannot be opened.
    - RuntimeError: If there is an error loading or aligning the Whisper model.

    The function converts the input file to .wav format and processes it using the Whisper speech recognition model to
    generate a time-aligned transcript. The transcript is then split into subtitles and combined into complete sentences
    where possible. The final transcript is written to the output file in .srt format.
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    srt_file = output_file
    print("creating audio file...")
    # Convert the input file
    # Export the audio to the .wav format
    # Load the original whisper model
    print("Standby while we load whisper")
    if device == "cuda":
        print(input_file)
        model = whisperx.load_model("medium.en", device)
        result = model.transcribe(input_file) 
        # Load the alignment model and metadata
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("aligning..............")
        # Align the whisper output
        result_aligned = whisperx.align(result["segments"], model_a, metadata, input_file, device)
        print(result_aligned.keys())
        srt_transcript = []
        i=1
        for (segment) in result_aligned["word_segments"]:
            start_time = srt.timedelta(seconds=int(segment['start']))
            end_time = srt.timedelta(seconds=int(segment['end']))
            text = segment['text'] #.strip().replace('-->', '->')
            srt_transcript.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))
            i+=1
        srt_text = (srt.make_legal_content(srt.compose(srt_transcript)))
        print("Written to srt text as words aligned by timecode")
        subs = list(srt.parse(srt_text))
        combined_subs = []
        i = 0
        while i < len(subs):
            sub = subs[i]
            start = sub.start.total_seconds()
            end = sub.end.total_seconds()
            text = sub.content
            iscomplete = False
            #minimum_length = 30 #default 30, set above
            count = 0
            j = i + 1
            #while there are more subs and the current sub is shorter than minimum duration, combine
            while (j < len(subs)) and ((end - start) < (int(minimum_length))):
                #print(j, "<", len(subs), "(i.e. not last sub), and, ", end, "-", start, "<", minimum_length, "second quotes minimum")
                addsub = subs[j]
                text += ' ' + addsub.content
                end = addsub.end.total_seconds() 
                j += 1
            #check if the sentence is complete
            last_three_characters = text[-3:]
            for char in last_three_characters:
              if char in ".?!:-":
                  iscomplete = True
            if (iscomplete == True) or (iscomplete == "True"):
                c.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?,?)", (start, end, text, srt_file,  "None", "None", "None")) #embedding, category, similarity
                combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
                count +=1
                i=j+1
            else:
              #  print("Sentance not complete, starting loop:")
                while (j<len(subs)):
                    if (j==len(subs)):
                        break
                    addsub = subs[j]
                    text += ' ' + addsub.content
                    end = addsub.end.total_seconds() 
                    #check if the sentence is complete
                    last_three_characters = text[-4:]
                    for char in last_three_characters:
                      if char in ".?!:-":
                        iscomplete = True
                    if (iscomplete == True):
                        #print (i, "-->", j)
                        break
                    j += 1
                c.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?,?)", (start, end, text, srt_file, "None", "None", "None")) #embedding, category, similarity
                combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
                count +=1
                i=j+1
        conn.commit()

        with open(srt_file, 'w+') as f:
              f.write(srt.make_legal_content(srt.compose(combined_subs)))
        f.close()
        print(srt.compose(combined_subs))
        print("written to file:", srt_file)
        conn.close()
        return srt_file

def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):
    unique_speakers = set()

    for seg in result_segments:
        wdf = seg['word-segments']
        if len(wdf['start'].dropna()) == 0:
            wdf['start'] = seg['start']
            wdf['end'] = seg['end']
        speakers = []
        for wdx, wrow in wdf.iterrows():
            if not np.isnan(wrow['start']):
                diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
                diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
                # remove no hit
                if not fill_nearest:
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                else:
                    dia_tmp = diarize_df
                if len(dia_tmp) == 0:
                    speaker = None
                else:
                    speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
            else:
                speaker = None
            speakers.append(speaker)
            unique_speakers.add(speaker)
        seg['word-segments']['speaker'] = speakers
        seg["speaker"] = pd.Series(speakers).value_counts().index[0]

    # create word level segments for .srt
    word_segments = []
    for seg in result_segments:
        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                word_segments.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": f"[{speaker}]: " + seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    # create chunk level segments split by new speaker
    chunk_segments = []
    for seg in result_segments:
        previous_speaker = None
        chunk_seg_text_start = None
        chunk_seg_text_end = None

        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                if previous_speaker is None:
                    previous_speaker = speaker

                if chunk_seg_text_start is None:
                    chunk_seg_text_start = int(wrow["segment-text-start"])

                if speaker != previous_speaker:
                    # This word must be new chunk
                    # Add all words until now as chunk
                    # Start new chunk
                    chunk_segments.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "speaker": seg["speaker"],
                            "text": seg["text"][chunk_seg_text_start:chunk_seg_text_end],
                        }
                    )
                    chunk_seg_text_start = int(wrow["segment-text-start"])
                    previous_speaker = speaker
                else: 
                    # This word is still good part of chunk
                    pass

                chunk_seg_text_end = int(wrow["segment-text-end"])

        if chunk_seg_text_end is not None:
            chunk_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text": seg["text"][chunk_seg_text_start:chunk_seg_text_end],
                }
            )

    return result_segments, word_segments, chunk_segments, unique_speakers

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker

def output_transcript(srt_file):
    f = open(srt_file,"r")
    srt_text = f.read()
    f.close()
    print(srt_file)
    subtitles = srt.parse(srt_text)
    # Parse the subtitles into list format
    # combine subs does not work???
    transcript = ""
    for sub in subtitles:
        transcript += sub.content
        # Insert into the database
    with open(str("transcribed_"+srt_file), 'w+') as f:
              f.write(transcript)
    f.close()
    return transcript #py_transcribed if use above

def find_new_codes(srt_file):
    print("Looking for new research codes! \n \n Try this \n \n")
    transcript = output_transcript(srt_file) #py_transcribed if use above
    print(transcript)
    # Segment size for sub-summarization. Default is 5 minutes. For videos with a lot of people speaking at once, or videos where the speaker(s) speak especially fast, you may want to reduce this.
    segment_size = 200000

    end_chars = ".?!;:"
    # Convert the transcript list object to plaintext so that we can use it with OpenAI
    transcript_segments = [[]]
    seg = ""
    transcript_index = 0
    last_cutoff = 0
    seglength = 0
    print("begin for line in transcript within researchcode summary")
    for line in transcript:
        # Add this line's text to the current transcript segment
        transcript_segments[transcript_index].append(line)
        for i in range(len(transcript_segments[transcript_index])):
            seg = ""
            lineseg = transcript_segments[transcript_index][i] #Line segment at this index,   #len(linesegs)
            seglength += len(lineseg.split()) #add the total len(linesegs) for your length
        # If this line is more than segment_size seconds after the last cutoff, then we need to create a new segment
        if (((seglength) > segment_size) and ((lineseg[-3:] in end_chars) )):#or (seglength>(1.5*segment_size))
            transcript_index += 1
            transcript_segments.append([])
            seglength = 1
    transegs = []
    for i in range(len(transcript_segments)):
        transcript_segments[i] = "".join(transcript_segments[i])
            # For each segment of the transcript, summarize
    transcript_segment_summaries = ""
    conversation = []
    badcodes=[]
    j=0
    instructions="Briefly revise, simplify, output concise keyphrases:"
    conversation.append({'role': 'system', 'content':instructions}) #+ transcript_segments[j]
    print(instructions)
    for i in range(len(transcript_segments)):
        transcript_segment = transcript_segments[i]
        print(transcript_segment) #TODO #WHY THEY ALL HAVE SPACES BETWEEN WORDS
        # Use the OpenAI Completion endpoint to summarize the transcript
        #model_id="gpt-3.5-turbo",
    
        conversation.append({'role': 'user', 'content': "'"+transcript_segment+"'"})
        response  = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= conversation,   #consise 
            #"Completely summarize the following text:\n"+transcript_segment +" \n",
            temperature=0.4,
            max_tokens=maxspend,
            top_p=1,
            frequency_penalty=2,
            presence_penalty=-2,
        )
        print("................................")
        newcode=str(response.choices[-1].message.content)
        delimiter_pattern = r"[,.;:\n]+"  # matches any of the separators
        phrases = re.split(delimiter_pattern, newcode)
        phrases = [phrase.strip() for phrase in phrases if phrase.strip()]  # remove leading/trailing spaces and empty strings
        for phrase in phrases:
         if (  (len(str(phrase))<8) or 250<(len(str(phrase))) or  ("Content" in phrase)  or ("phrase" in phrase) or ("Phrase" in phrase)  or  ("implify" in phrase) or  ("Content" in phrase) or  ("content" in phrase)  or ("ummary" in phrase)  or ("Text" in phrase) or ("text" in phrase) or ("TEXT" in phrase) or ("Quote" in phrase) or ("quote" in phrase) or ("Keywords" in phrase) or ("Keyword" in phrase) or  ("Sorry" in phrase) or ("sorry" in phrase) or ("text" in phrase) or ("summarize" in phrase)or ("Summary" in phrase)  or ("summary" in phrase) or ("Summarize" in phrase) or ("esearch" in phrase)):
            print("\nREJECTED: " + phrase)
         else:
            transegs.append(phrase)
            transcript_segment_summaries += str(","+phrase) 
    return(str(transcript_segment_summaries))

    if breakpoints:
        if breakpoints[len(breakpoints) - 1] < len(transcript_segment_summaries): #if the last value of breakingpoints[] is less than the last index of transcript_segment_summaries
            breakpoints.append(len(transcript_segment_summaries))#addbreak at end of transcript summaries
    else:
        breakpoints.append(len(transcript_segment_summaries))
    summary = ""
    generated_codes = ""
    for code in badcodes:
        summary += code
    if len(summary)>20:
        response = openai.Completion.create(
              model="text-curie-001",
              prompt=f"List specific keywords, or keyphrases, from the following: {summary}\n", 
              temperature=0.1,
              max_tokens=100,
              top_p=1,
              frequency_penalty=10,
              presence_penalty=0,
              )
        transcript_segments.append( str(response.choices[0].text.strip()))
    for seg in transcript_segments:
        generated_codes += str(","+ seg)
    return str(generated_codes +",")

    openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"List keywords for the following transcript segments: {summary}\n",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    lastbatch=0

    while (len(breakpoints)>0): # (breakpoints[len(breakpoints)-1] > 0)): #while the last index isn't 0
            batch = (breakpoints[0])
            prompt = ""
            for i in range(batch-lastbatch):
                prompt += str(i+1)+":\n"+transcript_segment_summaries[(lastbatch+i)]+"\n"    
            print("Prompt", i, " ", prompt)
            print("...")
            for each in (transcript_segment_summaries):
                summary += each
            # Use the OpenAI Completion endpoint to summarize the transcript
            newconvo.append({'role': 'system', 'content': 'Format a .csv: \n'})
            newconvo.append({'role': 'user', 'content' : summary})
            response  = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= newconvo,   #consise 
            #"Completely summarize the following text:\n"+transcript_segment +" \n",
            temperature=0.0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=2,
            presence_penalty=0,
        )
    print(summary)

    return summary


#def get_categories(input_string):
def get_categories(input_string: str, db_file) -> list[tuple[str, list[float]]]:
    """
    Given a string of comma-separated categories, retrieves the embeddings
    of each category using OpenAI's text-embedding-ada-002 model. Returns a
    list of tuples, where each tuple contains a category and its corresponding
    embedding as a list of floats.
    Args:
        input_string: A string of comma-separated categories.
    Returns:
        A list of category embeddings
         Stores in dbfile a list of tuples, where each tuple contains a category and its
        corresponding embedding as a list of floats.
    """
       # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    delimiter_pattern = r"[,.;:\n]+"  # matches any of the separators
    phrases = re.split(delimiter_pattern, input_string)
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]  # remove leading/trailing spaces and empty strings
    cat_list = phrases
    i=0
    category_embeddings = []
    for category in cat_list:
      # Get the embeddings for the query
      time.sleep(60/450)#delay in seconds
      print(".")
      response = openai.Embedding.create(model="text-embedding-ada-002", input=category)
      embedding=(response["data"][0]["embedding"])
      category_embeddings.append((category, embedding))
      c.execute("INSERT INTO categories VALUES (?,?,?,?)", (category,(json.dumps(embedding)),0,0))
      conn.commit()
    conn.close()
    return category_embeddings

    
# Split the input string into a list of values using the csv module

#note, categories is an array of tuples [(category, embedding)]
def categorize_codes(srt_file: str, categories: list[tuple[str, list[float]]], output_path: str, db_file: str) -> str:
    """
    Categorizes the subtitles in the specified srt_file into the provided categories and writes the output to an srt file.

    Args:
    - srt_file (str): A string representing the path to the srt file to be categorized.
    - categories (list[tuple[str, list[float]]]): A list of tuples representing the categories to be used. Each tuple contains a category name (str) and its corresponding embedding (List[float]).
    - output_path (str): A string representing the path to write the output categorized srt file.
    - db_file (str): A string representing the path to the database file used to store the subtitles.

    Returns:
    - A string representing the path of the output categorized srt file.
    """

    print("Loading subs from SRT to DB")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")#(db_file)
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, embedding TEXT, categorized TEXT, similarity_score REAL)")
    memconn.commit()

    c.execute("SELECT category,category_embedding FROM categories WHERE category_embedding != 'None'")
    DBcategories = c.fetchall()
    # Close the connection
    index = 0
    #setup category counter
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? AND embeddings != 'None'", (srt_file,))
    subtitles = c.fetchall()
    # Get the text of the subtitles 
    for time_start,time_end,text,sub_embedding in subtitles:
        memc.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?)", (time_start, time_end, text, sub_embedding,  "None", "None")) #embedding, category, similarity
        memconn.commit()
        newavg=0
        similarity = 0
        relevant_category = None
        relevant_embedding = None
        sub_embedding = json.loads(sub_embedding)
        categories = fetch_embedded_categories(db_file)
        for category, category_embedding in categories:
            category_embedding = json.loads(category_embedding)
           # Calculate the highest cosine similarity by category
            simscore = np.dot(category_embedding, sub_embedding) / (np.linalg.norm(category_embedding) * np.linalg.norm(sub_embedding))
            print(simscore)
            if simscore > similarity:
               similarity = simscore
               relevant_category = category
               relevant_embedding = category_embedding
            memc.execute("UPDATE subtitles SET categorized = ?, similarity_score = ? WHERE text = ?", (relevant_category, similarity, text))
            c.execute("UPDATE subtitles SET category = ?, similarity = ? WHERE text = ?", (relevant_category, similarity, text))
            memconn.commit()
            conn.commit()
        conn.commit()
        c.execute(" SELECT avgSimilarity FROM categories WHERE category = ?", (relevant_category,))
        avgSimilarity = c.fetchall()
    # Get the categorized results
    memc = memconn.cursor()
    # Get the categorized results
    memc.execute("SELECT start,end, categorized,similarity_score,text FROM subtitles") #ORDER BY similarity_score DESC LIMIT ?", (top_n,))
    results = memc.fetchall()
    memconn.close()#f.close() ##why
    # Print the results
    categorized_subs = []
    index = 1
    for time_start,time_end,categorized,similarity_score,sub_text in results:
        #for each categorized sub, count it in the tuple
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        # Print the results
        print(time_start, time_end, "\n", categorized, "==", similarity_score, "\n", sub_text)
        categorized_subtext =  str("     ~'" + categorized + "'==" + str(similarity_score) +  "% \n" + sub_text + "\n \n") # "\n".join([category_similarity_str, sub_text, "\n\n"])
        categorized_subs.append(srt.Subtitle(index=index, start=time_start, end=time_end, content=categorized_subtext))
        index+=1
    category_results = ""
    category_occurrences=fetch_category_counts(db_file)
    for i, (cat, qty) in enumerate(sorted(category_occurrences, key=lambda x: x[1], reverse=True)):
            category_results += str(str(qty) + ", " + cat + ",\n")
    prompted=""
    for code, embedding in DBcategories:
        prompted+=code+","
    #Add the query results to the top of the subtitle for easy reading of one single output file
    category_results_string = str("INPUT: \n File:" + input_file + "\n" + "Quote Length apx:" + str(minimum_length) + " Seconds, \n Categories Prompted: " + prompted + "\n\n RESULTS per Category: \n" + category_results + "\n \n \n")

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w') as f:
        f.write(category_results_string + "\n" + srt.make_legal_content(srt.compose(categorized_subs)))
    f.close()
    print(category_results_string)
    print("written final categorized subs to file")
    print(output_path)
    return output_path


def revise_categories(srt_file, new_file, db_file):
    categories = fetch_embedded_categories(db_file)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    catdeletions=""
    while catdeletions != "n":
        catdeletions=input("DELETE ANY CATEGORIES? type  'n' to exit , or category name to delete and refine categories")
        c.execute("DELETE FROM categories WHERE category = ?", (catdeletions,))
        conn.commit()
        categories = fetch_embedded_categories(db_file)
        #for cat,embed in categories:
            #print(cat)
        categorize_codes(srt_file, categories, new_file, db_file)

def fetch_embedded_subs(srt_file, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? AND embeddings != 'None'", (srt_file,))
    subtitles = c.fetchall()
    # Get the text of the subtitles
    return subtitles
def fetch_embedded_categories(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the categories from the database
    c.execute("SELECT category,category_embedding FROM categories")
    categories = c.fetchall()
    # Get the text of the subtitles
    with open("categories.csv", "w+") as f:
        for text, emb in categories:
         writer = csv.writer(f)
         writer.writerow(text)
    f.close()
    return categories

def fetch_all_subs(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles")
    subtitles = c.fetchall()
    # Get the text of the subtitles
    return subtitles
def fetch_category_counts(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT category, COUNT(*) FROM subtitles GROUP BY category')
    # Fetch all results and store in list of tuples
    category_counts = sorted(c.fetchall(), key=lambda x: x[1], reverse=True)
    return category_counts

if __name__ == "__main__": 
  db_file = "data.db"
  output_file =  os.path.splitext(os.path.basename(input_file))[0] + ".txt"
  m4a_to_srt_categorized(input_file, input_string, output_file, db_file)
