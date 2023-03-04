
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
