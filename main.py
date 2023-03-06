import os
import glob
import sys
import csv
import sqlite3
import numpy as np
import srt
import json
import ffmpeg
import whisper
import whisperx
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from pydub import AudioSegment
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
    input_file (str): The path of the input file .m4a
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

input_file = input("To begin, provide the input file:  ")
filename = os.path.splitext(os.path.basename(input_file))[0]
wav_output = filename + ".wav"
srt_output_path = filename + ".srt"
input_string = input("Provide a list of categories (or research codes separated by commas)")

minimum_length = 30 #quote length
minimum_length = input("minimum quote length?") # DEFAULT TO 30
  
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
    
def setup(input_file):
      db_setup(db_file)
    #if "m4a" in input_file:
      filename = os.path.splitext(os.path.basename(input_file))[0]
      wavsound = filename + ".wav"
      srt_filename = filename + ".srt"
      audio = AudioSegment.from_file(input_file, format="m4a")
   
        # Export the AudioSegment object as a .wav file
      audio.export(wavsound, format="wav")
      print(input_file)
      print(wavsound)
      return create_srt_transcript(wavsound, srt_filename, "cuda")

  
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
    srt_file = output_file
    input_audio = input_file
    print("creating audio file...")
    # Convert the input file
    #audio = AudioSegment.from_file(input_file, format=input_file.split(".")[-1])
    #audio.export(input_audio, format="wav")
    
    print("wav is compatible!")
    # Export the audio to the .wav format

    # Load the original whisper model
    print("Standby while we load whisper")
    if device == "cuda":
        print(input_file)
    #try:
        model = whisperx.load_model("large", device)# ("medium.en", device)
        result = model.transcribe(input_file) 
        # Load the alignment model and metadata
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("aligning timecode")
        # Align the whisper output
        result_aligned = whisperx.align(result["segments"], model_a, metadata, input_audio, device)

        # Create the .srt transcript
        srt_transcript = []
        i=1
        for (segment) in result_aligned["word_segments"]:

            start_time = srt.timedelta(seconds=int(segment['start']))
            end_time = srt.timedelta(seconds=int(segment['end']))

            text = segment['text'] #.strip().replace('-->', '->')
            srt_transcript.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))
            i+=1
        #return write_to_file((srt_transcript), input_subtitle_file)

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
            #while there are more subs and the current sub is shorter than minimum duration, combine
            while (j < len(subs)) and ((end - start) < (int(minimum_length))):
                #print(j, "<", len(subs), "(i.e. not last sub), and, ", end, "-", start, "<", minimum_length, "second quotes minimum")
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
                    if (iscomplete == True):
                        #print (i, "-->", j)
                        break
                    j += 1
                combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
                count +=1
                i=j+1
        with open(srt_file, 'w+') as f:
              f.write(srt.make_legal_content(srt.compose(combined_subs)))
        f.close()
        print(srt.compose(combined_subs))
        print("written to file:", srt_file)

        return srt_file
    #except Exception as e:
    #    print("Failed to align whisper output: %s", e)


def get_embeddings(srt_file, db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the subtitles from the database
    c.execute("SELECT start,end,text FROM subtitles WHERE srt_file = ? and embeddings='None'", (srt_file,))
    subtitles = c.fetchall()
    # Get the text of the subtitles
    for time_start,time_end,sub_text in subtitles:
        response = openai.Embedding.create(model="text-embedding-ada-002", input=sub_text)
        embedding = response["data"][0]["embedding"]
        c.execute("UPDATE subtitles SET embeddings = ? WHERE start = ? AND end = ? AND srt_file = ?", (json.dumps(embedding), time_start, time_end, srt_file))
        # Commit the changes
        conn.commit()
        # print(format_time(time_end))
    # Close the connection
    conn.close()
    print("Embedded meanings and associations found for every soundbite-sub.")
    print(" ...")

#def get_categories(input_string):
def get_categories(input_string: str) -> list[tuple[str, list[float]]]:
    """
    Given a string of comma-separated categories, retrieves the embeddings
    of each category using OpenAI's text-embedding-ada-002 model. Returns a
    list of tuples, where each tuple contains a category and its corresponding
    embedding as a list of floats.

    Args:
        input_string: A string of comma-separated categories.

    Returns:
        A list of tuples, where each tuple contains a category and its
        corresponding embedding as a list of floats.
    """

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
#def categorize(srt_file, categories, output_path, dbfile):
def categorize(srt_file: str, categories: list[tuple[str, list[float]]], output_path: str, db_file: str) -> str:
    """
    Categorizes the subtitles in the specified srt_file into the provided categories and writes the output to an srt file.

    Args:
    - srt_file (str): A string representing the path to the srt file to be categorized.
    - categories (List[Tuple[str, List[float]]]): A list of tuples representing the categories to be used. Each tuple contains a category name (str) and its corresponding embedding (List[float]).
    - output_path (str): A string representing the path to write the output categorized srt file.
    - db_file (str): A string representing the path to the database file used to store the subtitles.

    Returns:
    - A string representing the path of the output categorized srt file.
    """
    print("Loading subs")
    load_subtitles(srt_file, db_file)
    print("getting embeddings")
    get_embeddings(srt_file, db_file)
      #put subtitles in a new database

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")#(db_file)
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, categorized TEXT, similarity_score REAL, text TEXT)")
    memconn.commit()
          
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? and embeddings != 'None'", (srt_file,))
    subtitles = c.fetchall()
    # Close the connection
    conn.close()

    #setup category counter
    category_counter=[]
    for category, embedding in categories:
        category_counter.append((category, 0))
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
    #memconn.close()#f.close() ##why
    #memconn = sqlite3.connect(":memory:")##why would it break without closing and reopening memconn?
    memc = memconn.cursor()
    # Get the categorized results
    memc.execute("SELECT start,end,categorized,similarity_score,text FROM subtitles") #ORDER BY similarity_score DESC LIMIT ?", (top_n,))
    results = memc.fetchall()
    # Print the results
    categorized_subs = []
    index = 1
    for time_start,time_end,categorized,similarity_score,sub_text in results:
        #for each categorized sub, count it in the tuple
        for i, (cat, qty) in enumerate(category_counter):
          if categorized == cat: 
            category_counter[i] = (cat, qty + 1)
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        # Print the results
        #category_similarity_str = str("CATEGORY:'" + categorized + "'==" + (similarity_score) + "% ")
        print(time_start, time_end, "\n", categorized, "==", similarity_score, "\n", sub_text)
        categorized_subtext =  str("    == '" + categorized + "', %" + str(similarity_score) +  "\n" + sub_text + "\n \n") # "\n".join([category_similarity_str, sub_text, "\n\n"])
        categorized_subs.append(srt.Subtitle(index=index, start=time_start, end=time_end, content=categorized_subtext))
        index+=1
    print("Categories Parsed for:")
    #    print(input_string)
    print(category_counter)
    categoryct_txt = "categoryresults_" + os.path.splitext(os.path.basename(input_file))[0] + ".txt"

    with open(categoryct_txt, 'w+') as f:
        for i, (cat, qty) in enumerate(category_counter)
            f.write(str(str(qty) + " " + cat + "\n"))
    print(categoryct_txt)

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w+') as f:
        f.write(srt.make_legal_content(srt.compose(categorized_subs)))
    f.close()
    print("written final categorized subs to file")
    print(output_path)
    return output_path
      
def m4a_to_srt_categorized(input_file, input_string):
  transcribed_subtitles = setup(input_file)
  print(transcribed_subtitles)
  print("getting embeddings for categories: " + input_string)
  categories = get_categories(input_string)
  outfilename = "categorized_" + os.path.splitext(os.path.basename(input_file))[0] + ".srt"
  categorize(transcribed_subtitles, categories, outfilename, db_file)

if __name__ == "__main__":
  m4a_to_srt_categorized(input_file, input_string)
