from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Get the set of English stopwords to remove from reviews
stopset = set(stopwords.words('english'))
# Do not include 'not' as a stop word because it can have a strong impact on the meaning of a sentence
stopset.remove('not')

# Converts the text file of amazon review data to a cleaned csv file
def convert_txt_to_csv(input_file_name, output_file_name):
    # Create a tokenizer that accepts only alphabetic words and periods and exclamation points
    tokenizer = RegexpTokenizer("[a-zA-Z]+|\.|!")
    with open(input_file_name, "r") as input_file:
        with open(output_file_name, "w") as output_file:
            line_num = 0
            rating = ""
            for line in input_file:
                line_num += 1
                # Get the ratings
                if line_num == 1:
                    # Use only the number of stars
                    rating = line[8:9]
                # Get the review text
                elif line_num == 8:
                    # Remove the label and convert to lower case
                    line = line[8:-1].lower()
                    # Split line into tokens
                    line = tokenizer.tokenize(line)
                    # Remove stop words
                    line = [i for i in line if i not in stopset]
                    # Combine lines back into new string
                    new_line = ' '.join(line)
                    output_file.write(new_line + ',' + rating + '\n')
                    line_num = 0

convert_txt_to_csv("../../Downloads/amazon_total.txt", "amazon_total.csv")