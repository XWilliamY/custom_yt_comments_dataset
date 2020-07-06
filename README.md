# Sentiment Analysis on Blackpink's How You Like That MV


You should first set up a virtual environment. You can create a Python 3 virtual environment by entering the following command:
``` python3 -m venv collect_comments ```

collect_comments can be any name, it'll just be the name of your virtual environment. To activate it, run:

``` source collect_comments/bin/activate ```

Next, install all required libraries:
``` pip3 install requirements.txt ```

Now you can collect comments from a specific video url with the following command:
```
python3 get_comments_of_video_id.py --order time --csv_filename csv_filename --apikey path/to/apikey.json
```

Additional arguments include:

| Command | Description |
|-|-|
| --part | Desired part of commentThread: id, snippet, and/or replies. Enter comma-separated string with no spaces. |
| --maxResults | How many comments to "display" per API call |
| --write_lbl | Whether to save comments to csv line-by-line as comments are sourced |
| --csv_filename | Name of csv file that comments should be saved to |
| --token_filename | This script also saves each pageToken: --token_filename will save these tokens to the specified filename |
| --video_url | Link to target YouTube video |
| --order | Order comments by time or by relevance. The default YouTube sets is to time |


TODO:
- [ ] Provide a way for users to give a list of desired features from Data API

