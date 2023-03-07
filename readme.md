# Coded Transcripts from .m4a and category prompts
## for qualitative research using large quantities of audio

This Python script attempts to automate the research process for interview transcripts. There is an error rate on autotranscribe that could be improved with the Whisper API, or locally with 10gb vram.

.m4a audio files are auto-transcribed with Whisper, each subtitle text-embedding is scored by similarity to a list of user-defined prompts, and every spoken sentence is categorized in the resulting .SRT file. The ADA-002-Embedding AI model seems quite accurate with short phrases but the output will need human verification for accuracy.

See example podcast output: ![]()
https://github.com/thegoodwei/Audio_SortbySub/blob/main/example/philosophy-of-meditation-vervaeke.txt

#### Instructions Below:

![/example/philosophy-of-meditation-vervaeke.txt](icon.svg)

### Methodology:

Automatic speech recognition (ASR): We use the PyDub library to convert the audio files to mono WAV format, and then pass them through the Whisper automatic transcription, using WhisperX for word-by-word timecode alignment. The resulting .srt subtitles are saved as JSON.

Category definition: Users define categories of interest and provide example phrases or sentences that represent each category. These example phrases are used to create category embeddings using OpenAI's text-embedding-ada-002 model.

Category embedding: We use the OpenAI text-embedding-ada-002 model to embed each category in a high-dimensional space. The resulting embeddings are saved to a CSV file.

Categorization: We use the text-embedding-ada-002 model and calculate the cosine similarity between each sentence in the autotranscribed file and the embeddings of the user-defined categories. The sentence is then assigned to the category with the highest similarity score. The resulting categorized subtitles are output as an SRT file.

Human Verification: We can utilize the SRT file format to enable easy verification of the categorization. Researchers can use video players that support SRT files to play the video with the categorized subtitles for quick editing. The video player can display the subtitles alongside the audio, allowing for quick verification of the categorization. If the categorization is incorrect, researchers can quickly fix the error and move to the next section. To add a new code, we can refine the list of categories to provide more phrases or sentences that represent the new code, and rerun the script to include those in the remaining unverified content.

Training a model: To ensure the accuracy of the categorization, researchers can also adust parameters such as the minimum quote length or quote breaking pattern as well as the similarity threshhold for assigning a sentence to any category.  Given enough data, researchers might choose to adjust the pretrained model in use or finetune a custom AI model to code transcripts with this specific dataset, as to compare manually coded transcripts to auto-categorized results, to build a bigger library of embeddings that might help the AI re-define each research-code or category as based on the subtext referenced. By implementing these systems, we can improve the efficiency and accuracy of the code-categorization process, ultimately enabling researchers to more effectively analyze extremely vast quantities of audio data for research.

## Installation
This script requires Python 3.7 or later, and the following packages:

tensorflow==2.6.0
tensorflow_hub==0.12.0
pysrt==1.1.2

  sudo apt-get install python3 git 
  python3 -m pip install  os glob csv sqlite3 numpy openai srt json whisper openai numpy pytorch wheel moviepy srt pydub  git+https://github.com/m-bain/whisperx.git


# Usage
### Setting up Categories
The script requires the user to define categories for the subtitles. To do this, you need to call the get_categories() function and provide a comma-separated string of categories. The function will retrieve the embeddings of each category using OpenAI's text-embedding-ada-002 model and return a list of tuples, where each tuple contains a category and its corresponding embedding as a list of floats.

  from categorizer import get_categories

  categories = get_categories("Category 1, Category 2, Category 3")


### Categorizing Subtitles
Once you have defined your categories, you can use the categorize() function to categorize subtitles. This function takes the following arguments:

srt_file: A string representing the path to the .srt file to be categorized.
categories: A list of tuples representing the categories to be used. Each tuple contains a category name (str) and its corresponding embedding (List[float]).
output_path: A string representing the path to write the output categorized srt file.
db_file: A string representing the path to the database file used to store the subtitles.
python

  from categorizer import categorize
  input_file = "subtitles.srt"
  output_file = "output.srt"
  database_file = "subtitles_database.db"

  categorize(input_file, categories, output_file, database_file)

The categorize() function will read in the .srt file and categorize each subtitle according to the closest category embedding. The output categorized .srt file will be saved to the path specified in output_path. The function returns a string representing the path of the output categorized srt file.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to verify results. This script has not been extensively bug-tested yet.
