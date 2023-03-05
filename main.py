from pydub import AudioSegment
import os
import glob
import sys
import csv
from custom_transcription_embedding_library import db_setup, load_subtitles, get_embeddings

db_file = "data.db"


# Set the path of the input folder containing .m4a files
input_folder = input("/path/to/folder")
def setup(input_folder)
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
      
      create_srt_transcript(input_file: filepath, output_file: output_path, device: str = "cuda")
  
  
def create_srt_transcript(input_file: str, device: str = "cuda"):
    """
    Create an srt transcript from an audio file.
    Args:
    - input_file (str): the path to the input audio file
    - output_file (str): the path to the output srt file
    - device (str): the device to use for processing, either "cuda" or "cpu" (default "cuda")
    Returns:
    None
    """
    input_audio = input_file
    #there used to be an 'with open' to create the file but I deleted it because errors 40 lines following btu it didnt fix
    print("creating audio file...")
    # input_audio = input_file[:-4] + ".wav"
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
    f.close()
    print("Written to srt text as words aligned by timecode")

    subs = list(srt.parse(srt_text))
    combined_subs = []
    i = 0
    while i < len(subs):
        sub = subs[i]
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        text = sub.content
        iscomplete = false
        minimum_length = 30
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
                iscomplete = find_complete_section(text)
                print("iscomplete looped value: ", str(iscomplete))
                if (iscomplete == True) or (iscomplete == "True"): # ('true' in iscomplete)):
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


  
def get_categories():

  # Prompt the user for a list of categories
  input_string = input("Provide a list of categories (or research codes), separated by commas. This program will auto-categorize each section of subtitle by the most similar embeddings: ")

  # Parse the input string into a list of categories
  word_list = input_string.split(",")

  # Write the list to a CSV file
  with open("categories.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerow(word_list)
    f.close()
  return word_list

      
    
# Split the input string into a list of values using the csv module

def categorize(input_folder, categorize[]):
  counter = 0
  for filepath in glob.glob(os.path.join(input_folder, "*.srt")):
      counter +=1
      filename = os.path.splitext(os.path.basename(filepath))[0]
      output_path = os.path.join(input_folder, "coded_" + filename + ".srt")
      load_subtitles(filepath, db_file)
      #TODO 
      #create list of embeddings for each category
      #for each subtitle section
        #calculate similarity of subtitle section against each category
        #calculate category with the greatest similarity score
        #write the most similiar category name under the timecode of the .srt text
        
      
      
      
      
      
      
      

      
      

def search_database(srt_file, db_file, query, top_n=int(total_categories)):
    print("Search has begun:") 
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, similarity_score REAL)")
    memconn.commit()
    # Get the embeddings for the query
    response = openai.Embedding.create(model="text-embedding-ada-002", input=query)
    query_embedding = response["data"][0]["embedding"]
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? and embeddings != 'None'", (srt_file,))
    subtitles = c.fetchall()
    # Close the connection
    conn.close()
    # Get the text of the subtitles 
    for time_start,time_end,sub_text,sub_embedding in subtitles:
        delayed_completion(delay_in_seconds=delay)
        # Get the embedding for the subtitle  
        sub_embedding = json.loads(sub_embedding)
        # Calculate the cosine similarity
        similarity = np.dot(query_embedding, sub_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(sub_embedding))
        # Print above avg results 
        # Insert data into the temporary database
        memc.execute("INSERT INTO subtitles VALUES (?,?,?,?)", (time_start, time_end, sub_text, similarity))
        memconn.commit()
    f.close()
    print("..........................")
    print(".......................................")
    print(".....................................................")
    # Get the top n results
    memc.execute("SELECT start,end,text,similarity_score FROM subtitles ORDER BY similarity_score DESC LIMIT ?", (top_n,)) #was ORDER BY similarity_score
    results = memc.fetchall()
    # Print the results
    selected_subs = []
    index = 1
    for time_start,time_end,sub_text,similarity_score in results:
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        #sel content = sub_text # "{similarity_score} \n {sub_text}"
        #selectofsub = find_complete_section(sub_text, user_prompt)
        # Print the results
        print(time_start, time_end, "\n", sub_text, "\n", "=", similarity_score)
        selected_subs.append(srt.Subtitle(index=index, start=time_start, end=time_end, content=sub_text))
        index+=1
    #Print selects to file, with details to terminal
    write_to_file(selected_subs, srt_file)

    return(srt_file)
      
      
      
      
      
      
      
if __name__ == "__main__":
  #batch m4a into timecode-aligned wordsubs, combined to end on complete sentences <30 seconds, saved as .srt files
  setup(input_folder)
  categories = get_categories()
  categorize(input_folder, categories)
  
