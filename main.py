from pydub import AudioSegment
import os
import glob
import sys
import csv
import sqlite3
import numpy as np
import openai
import srt
import json
import whisperx
from custom_library import db_setup, load_subtitles, get_embeddings
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

    write_to_file(newsubs, srt_file):
        This function writes the subtitles to a file in the SRT format.

    load_subtitles(srt_file, db_file):
        This function parses the SRT file and stores the subtitles in the SQLite database.

    setup(input_folder):
        This function prepares the input files by converting the .m4a files to .wav format and transcribing them to SRT format.

    create_srt_transcript(input_file, output_file, device):
        This function performs audio transcription using OpenAI's Whisper model and creates an SRT file.

Input:
    db_file (str): The path of the SQLite database file to store the subtitle data.
    input_folder (str): The path of the input folder containing .m4a files.
    input_string (str): A list of categories (or research codes separated by commas).
    minimum_length (int): The minimum length of a subtitle in seconds.

Output:
    Subtitle data is stored in the SQLite database and then categorized SRT files are generated for each input .m4a file.

"""

db_file = "data.db"
# Set the path of the input folder containing .m4a files
  # Prompt the user for a list of categories
print("This script will auto-transcribe full sentences of >30 seconds, and then categorize each section of subtitle by the most similar embeddings to the categories provided. ")
print("\n \n Input a file directory, and the program will convert each .m4a file into a .wav, for OpenAI Whisper to automatically transcribe locally. The WhisperX Module will align timecode for each word as an .srt file")
print("the minimum duration of each section is set to combine subs to at least 30 seconds and then end on an end-of-sentence break marker such as .!?,;:--.")
print("OpenAI ADA model generates embeddings (a vector of associated and related meanings) for each subtitle, as well as embeddings for each category/research code provided.")
print("We use cosign similarity comparison and assume the most similar category to be the most relevant tag for each section of text \n \n")
input_folder = input("To begin, provide the /path/to/inputfolder")

input_string = input("Provide a list of categories (or research codes separated by commas)")

minimum_length = 30

  
def db_setup(db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create the table if it doesn't exist
    c.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, srt_file TEXT, summary TEXT, embeddings TEXT)")
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()
    
def write_to_file(newsubs, srt_file):#="temp.srt"):
    #clear files to write project outputs
    with open(srt_file, 'w+') as f:
        f.write(srt.make_legal_content(srt.compose(newsubs)))
    f.close()
    print("written to file")
    return srt_file

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
        c.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?)", (start, end, text, srt_file, "None", "None"))
        #print(text)
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()
    

def setup(input_folder):
  # Iterate through the files in the input folder
  for filepath in glob.glob(os.path.join(input_folder, "*.m4a")):
      # Get the filename and path of the input .m4a file
      filename = os.path.splitext(os.path.basename(filepath))[0]
      output_path = os.path.join(input_folder, filename + ".wav")
    
      # Load the audio file into a AudioSegment object
      audio = AudioSegment.from_file(filepath, format="m4a")

      # Export the AudioSegment object as a .wav file
      audio.export(output_path, format="wav")

      # Remove the input .m4a files
      os.remove(filepath)
  db_setup(db_file)

#transcribe each wav file into an SRT
  for filepath in glob.glob(os.path.join(input_folder, "*.wav")):
      filename = os.path.splitext(os.path.basename(filepath))[0]
      output_path = os.path.join(input_folder, filename + ".srt")
      
      create_srt_transcript(filepath, output_path, "cuda")
  
  
def create_srt_transcript(input_file: str, output_file: str, device: str = "cuda"):
    """
    Create an srt transcript from an audio file.
    Args:
    - input_file (str): the path to the input audio file
    - output_file (str): the path to the output srt file
    - device (str): the device to use for processing, either "cuda" or "cpu" (default "cuda")
    Returns:
    None
    """
    srt_file = output_file
    input_audio = input_file
    print("creating audio file...")
    # Convert the input file
    audio = AudioSegment.from_file(input_file, format=input_file.split(".")[-1])
    audio.export(input_audio, format="wav")
    
    print("wav is compatible!")
    # Export the audio to the .wav format

    # Load the original whisper model
    print("Standby while we load whisper")
    try:
        model = whisperx.load_model("medium", device)
        result = model.transcribe(input_audio) #was input_audio if converted filetype
        # Load the alignment model and metadata
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("aligning timecode")
        # Align the whisper output
        result_aligned = whisperx.align(result["segments"], model_a, metadata, input_audio, device)
    except Exception as e:
        logging.error("Failed to align whisper output: %s", e)
        return
    #print(result["segments"]) # before alignment
    #print("_______word_segments")
    #print(result_aligned["word_segments"]) # after alignment
    #write_srt(result_aligned[segment], TextIO.txt)

  #    audio_basename = Path(audio_path).stem 
  #    with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt: 
  #        write_srt(result_aligned["segments"], file=srt)

    # Create the .srt transcript
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
    print(srt_text)
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
        #while there are more subs and the current sub is shorter than 30 seconds, combine
        while (j < len(subs)) and ((end - start) < (minimum_length)):
            addsub = subs[j]
            text += ' ' + addsub.content
            end = addsub.end.total_seconds() 
            j += 1
        #check if the sentence is complete
        last_three_characters = text[-4:]
        for char in last_three_characters:
          if char in ".,?!:-":
              iscomplete = True
        if (iscomplete == True) or (iscomplete == "True"):
            combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
            count +=1
            i=j+1
        else:
          #  print("Sentance not complete, starting loop:")
            while (j<len(subs)):
                print (i, "-->", j)
                if (j==len(subs)):
                    break
                addsub = subs[j]
                text += ' ' + addsub.content
                end = addsub.end.total_seconds() 
                #check if the sentence is complete
                last_three_characters = text[-4:]
                for char in last_three_characters:
                  if char in ".,?!:-":
                    iscomplete = True
                print("iscomplete looped value: ", str(iscomplete))
                if (iscomplete == True):
                    break
                j += 1
            combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
            count +=1
            i=j+1
    with open(srt_file, 'w+') as f:
          f.write(srt.make_legal_content(srt.compose(combined_subs)))
    f.close()
    print("written to file")
    return srt_file

def get_categories(input_string):
  # Parse the input string into a list of categories, separated by commas
  cat_list = input_string.split(",")

  # Write the list to a CSV file
  with open("categories.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerow(cat_list)
    f.close()
    
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, similarity_score REAL)")
    memconn.commit()
    
  category_embeddings = []
  for category in cat_list:
    # Get the embeddings for the query
    response = openai.Embedding.create(model="text-embedding-ada-002", input=category)
    category_embeddings.append((category, response["data"][0]["embedding"]))
  conn.close()

      
  return category_embeddings

    
# Split the input string into a list of values using the csv module

#note, categories is an array of tuples [(category, embedding)]
def categorize(srt_file, categories, output_path, dbfile):
      #put subtitles in a new database
      load_subtitles(srt_file, db_file)

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, categorized TEXT, similarity_score REAL, text TEXT)")
    memconn.commit()
          
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? and embeddings != 'None'", (srt_file,))
    subtitles = c.fetchall()
    # Close the connection
    conn.close()
    # Get the text of the subtitles 
    for time_start,time_end,sub_text,sub_embedding in subtitles:
        # Get the embedding for the subtitle, reset category similariy between each section
        sub_embedding = json.loads(sub_embedding)
        similarity = 0
        relevant_category = None

        for category, category_embedding in categories:
        
           # Calculate the highest cosine similarity by category
            simscore = np.dot(category_embedding, sub_embedding) / (np.linalg.norm(category_embedding) * np.linalg.norm(sub_embedding))
            if simscore > similarity:
               similarity = simscore
               relevant_category = category
        
        # Insert data into the temporary database
        memc.execute("INSERT INTO subtitles VALUES (?,?,?,?,?)", (time_start, time_end,  relevant_category, similarity, sub_text))
        memconn.commit()
    memconn.close()#f.close() ##why
    memconn = sqlite3.connect(":memory:")##why would it break without closing and reopening memconn?
    memc = memconn.cursor()
    # Get the categorized results
    memc.execute("SELECT start,end,categorized,similarity_score,text FROM subtitles") #ORDER BY similarity_score DESC LIMIT ?", (top_n,))
    results = memc.fetchall()
    # Print the results
    categorized_subs = []
    index = 1
    for time_start,time_end,categorized,similarity_score,sub_text in results:
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        # Print the results
        print(time_start, time_end, "\n", categorized, "==", similarity_score, "\n", sub_text)
        categorized_subtext = "\n".join([categorized, "=", str(similarity_score), "\n", sub_text])
        categorized_subs.append(srt.Subtitle(index=index, start=time_start, end=time_end, content=categorized_subtext))
        index+=1

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w+') as f:
        f.write(srt.make_legal_content(srt.compose(categorized_subs)))
    f.close()
    print("written final categorized subs to file")
    print(output_path)
    return output_path
      
      
if __name__ == "__main__":
  #batch m4a into timecode-aligned wordsubs, combined to end on complete sentences as .srt files
  setup(input_folder)
  #get the user input csv of categories, 
  categories = get_categories(input_string)
  #batch the whole folder .srt files 
  counter = 0
  for filepath in glob.glob(os.path.join(input_folder, "*.srt")):
      counter +=1
      db_file = "data" + counter + ".db"
      filename = os.path.splitext(os.path.basename(filepath))[0]
      output_path = os.path.join(input_folder, "codecategorized_" + filename + ".txt")
      categorize(filepath, categories, output_path, counter)
  
