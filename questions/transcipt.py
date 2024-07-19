from datasets import load_dataset, Dataset
import json
import os
from pathlib import Path
from typing import List
import sys


class TransciptDataset:
    def __init__(self, filepath: Path, savepath: Path):
        """Dataset class to handle transcript information from interview data.

        TranscriptDataset stores the metadata as lists of strings and stores the questions,
        answers, and named entities as lists of lists.

        Attributes:
            file: transcipt file location
            title: podcast episode title
            subtitle: podcast episode subtitle
            description: podcast episode subtitle TODO: fix newline parsing
            host: identify host full name (first and last) as string
            guest: identify host full name (first and last) as string
            named_entities: TBD
            questions: list of paragraphs from host, to be filtered to questions only
            answers: list of paragraphs from guest, to be filtered to answers only
        """
        self.file: List[str] = []
        self.title: List[str] = []
        self.subtitle: List[str] = []
        self.description: List[str] = []
        self.host: List[str] = []
        self.guest: List[str] = []
        # self.named_entities: List[list] = []
        self.questions: List[list] = []
        self.answers: List[list] = []
        
        files = os.listdir(filepath)
        for file in files:
            self.parse_file(filepath.joinpath(file))
            self.file.append(file)

        self.to_json_by_episode(savepath)
    

    def parse_file(self, file: str):
        """Parse a transcript text file and store its data in TranscriptDataset attributes.

        This function extracts the key information in the heading of the transcript like
        the title, guest, host, etc. and stores it in the attribute lists. The back-and-forth
        of the interview is captured as questions (spoken by host) and answers (spoken by 
        guest) such that the number of entires in each list are equivalent. This will allow
        analysis of answer characteristics based on question characteristics.

        Args:
            file: the transcript full file path to load and process.
        """
        with open(file) as f:
            lines = f.readlines()
            guest_labeled = False
            curr_speaker = None
            conversation_started = False
            curr_dialogue = ""
            host_questions: List[str] = []
            guest_answers: List[str] = []

            for line in lines:

                # Extract header information about transcript.
                if not guest_labeled:
                    if 'TITLE:' in line and 'SUBTITLE:' not in line:
                        self.title.append(line.split('TITLE: ')[-1].split('\n')[0])
                    elif 'SUBTITLE:' in line:
                        self.subtitle.append(line.split('SUBTITLE: ')[-1].split('\n')[0])
                    elif 'DESCRIPTION:' in line:
                        self.description.append(line.split('DESCRIPTION: ')[-1].split('\n')[0])
                    elif 'Host:' in line:
                        host_firstname, host_lastname = line.split(' ')[1:3]
                        host_fullname = host_firstname + ' ' + host_lastname
                        self.host.append(host_fullname)
                    elif 'Guest:' in line:
                        guest_firstname, guest_lastname = line.split(' ')[1:3]
                        guest_fullname = guest_firstname + ' ' + guest_lastname
                        self.guest.append(guest_fullname)
                        curr_speaker = guest_fullname
                        guest_labeled = True

                # Filter out unnecessary annotations.
                if 'music\n' in line:
                    continue
                if line in ['\t', '\n', ' \n', 'music\n']:
                    continue
                if ('[' in line) and (']' in line):
                    continue

                # Once the header information is extracted (the last step being identifying the guest), the
                # dialogue is chunked by transcript speaker labels such that one chunk is designated as the 
                # text between the host's identifier and the guests identifier. 
                if guest_labeled:
                    if ((guest_fullname + ':' in line) or (guest_firstname + ':' in line) or ('OUTRO' in line)) and ('Guest:' not in line):
                        # If guest identifier found and host is current speaker, save dialogue and reset buffer. 
                        # Otherwise, the host may be talking twice and the dialogue list should just be extended.
                        if curr_speaker == host_fullname:
                            if len(curr_dialogue) > 0:
                                host_questions.append(curr_dialogue)
                            del curr_dialogue
                            curr_dialogue = ""
                        curr_dialogue += line.split(guest_fullname + ': ')[-1]
                        curr_speaker = guest_fullname

                    elif (host_fullname + ':' in line) or (host_firstname + ':' in line) or ('INTRO' in line) or ('OUTRO' in line):
                        if curr_speaker == guest_fullname:
                            if len(curr_dialogue) > 0:
                                guest_answers.append(curr_dialogue)
                            del curr_dialogue
                            curr_dialogue = ""
                        curr_dialogue += line.split(host_fullname + ': ')[-1]
                        curr_speaker = host_fullname
                        conversation_started = True
                
                    elif conversation_started:
                        curr_dialogue += line

        # Ensure length of questions and answers is the same.
        if len(host_questions) > len(guest_answers):
            if len(host_questions) - len(guest_answers) == 1:
                host_questions.pop(len(host_questions)-1)
            else:
                sys.exit(f'Host Questions Length: {len(host_questions)}, Guest Answers Length: {len(guest_answers)}!')
        self.questions.append(host_questions)
        self.answers.append(guest_answers)


    def to_json_by_episode(self, savepath: Path):
        """Convert an entry-keyed transcript-extracted dictionary to a json file.

        Args:
            savepath: file path to save a 'transcript_data.json' file to.
        """
        transcript_dict = {}
        for idx in range(len(self.title)):
            entry_dict = {}
            entry_dict['file'] = self.file[idx]
            entry_dict['title'] = self.title[idx]
            entry_dict['subtitle'] = self.subtitle[idx]
            entry_dict['description'] = self.description[idx]
            entry_dict['host'] = self.host[idx]
            entry_dict['guest'] = self.guest[idx]
            entry_dict['questions'] = self.questions[idx]
            entry_dict['answers'] = self.answers[idx]
            transcript_dict[idx] = entry_dict
        
        formatted_dict = json.dumps(transcript_dict)
        formatted_dict = json.loads(formatted_dict)
        savefile = savepath.joinpath('transcript_data.json')
        with open(savefile, 'w', encoding='utf-8') as f:
            json.dump(formatted_dict, f, ensure_ascii=False, indent=4)



    def to_json_by_attribute(self, savepath: Path):
        """Convert an attribute-keyed transcript-extracted dictionary to a json file.

        Args:
            savepath: file path to save a 'transcript_data.json' file to.
        """
        transcript_dict = {}        
        transcript_dict['file'] = self.file
        transcript_dict['title'] = self.title
        transcript_dict['subtitle'] = self.subtitle
        transcript_dict['description'] = self.description
        transcript_dict['host'] = self.host
        transcript_dict['guest'] = self.guest
        transcript_dict['questions'] = self.questions
        transcript_dict['answers'] = self.answers

        formatted_dict = json.dumps(transcript_dict)
        formatted_dict = json.loads(formatted_dict)
        savefile = savepath.joinpath('transcript_data.json')
        with open(savefile, 'w', encoding='utf-8') as f:
            json.dump(formatted_dict, f, ensure_ascii=False, indent=4)
