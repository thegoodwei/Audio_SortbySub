# Deductive Autocoding, 
## Autotranscription and Auto Categorization for Qualitative Data Analysis with .m4a audio
This Python script is designed to automate the deductive coding of autotranscribed .m4a files for qualitative research analysis. The script uses OpenAI's text-embedding-ada-002 model to categorize subtitles into user-defined categories and writes the output to an .srt file.

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

