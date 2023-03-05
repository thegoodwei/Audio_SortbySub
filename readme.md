# Deductive Autocoding
### Autotranscription and Auto Categorization for Qualitative Data Analysis with .m4a audio

This Python script is designed to automate the deductive coding of autotranscribed .m4a files for qualitative research analysis. The script uses OpenAI's text-embedding-ada-002 model to categorize subtitles into user-defined categories and writes the output to an .srt file. 

#### Purpose
Qualitative research often involves the analysis of large amounts of audio data, which can be time-consuming and resource-intensive. This project presents an automated approach to categorizing autotranscribed audio files using OpenAI's text-embedding-ada-002 model. The approach involves converting audio files to text using automatic speech recognition (ASR) software, and then using the resulting text to categorize the audio based on user-defined categories. The resulting categorized subtitles are output as an SRT file, which can be used for further analysis. The script is in Beta and this approach will need be tested with results manually verified for acccuracy.

### Methodology:
Our approach involves several steps:

Automatic speech recognition (ASR): We use the PyDub library to convert the audio files to mono WAV format, and then pass them through the Google Cloud Speech-to-Text API for automatic transcription. The resulting text is then saved as a JSON file.

Category definition: Users define categories of interest and provide example phrases or sentences that represent each category. These example phrases are used to create category embeddings using OpenAI's text-embedding-ada-002 model.

Category embedding: We use the OpenAI text-embedding-ada-002 model to embed each category in a high-dimensional space. The resulting embeddings are saved to a CSV file.

Categorization: We use the text-embedding-ada-002 model to calculate the cosine similarity between each sentence in the autotranscribed file and the embeddings of the user-defined categories. The sentence is then assigned to the category with the highest similarity score. The resulting categorized subtitles are output as an SRT file.

Human Verification: We can utilize the SRT file format to enable easy verification of the categorization. Researchers can use video players that support SRT files to play the video with the categorized subtitles for quick editing. The video player can display the subtitles alongside the audio, allowing for quick verification of the categorization. If the categorization is incorrect, we can quickly fix the error and move to the next section. To add a new code, we can refine the list of categories to provide more phrases or sentences that represent the new code, and rerun the script to include those in the remaining unverified content.

Training a model: To ensure the accuracy of the categorization, researchers can compare manually coded transcripts to auto-categorized results, and the manual categorization can then be compared to the output of the script. Any discrepancies can be used to identify areas for improvement in the script. Researchers can adust parameters such as the minimum quote length or quote breaking pattern as well as the similarity threshhold for assigning a sentence to any category.  Given enough data, researchers might choose to adjust the pretrained model in use or train a custom AI model to code transcripts with their specific dataset. By implementing these systems, we can improve the efficiency and accuracy of the code-categorization process, ultimately enabling researchers to more effectively analyze extremely vast quantities of audio data for research.


## Installation
This script requires Python 3.7 or later, and the following packages:

tensorflow==2.6.0
tensorflow_hub==0.12.0
pysrt==1.1.2
To install the required packages, run the following command:
sudo apt-get install python3 git
python3 -m pip install  os glob csv sqlite3 numpy openai srt json whisperx
 openai numpy pytorch wheel moviepy srt pydub  git+https://github.com/m-bain/whisperx.git


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

Please make sure to update tests as appropriate.

License
MIT

