import argparse
import random
import requests
import wikipedia
import xml
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from time import sleep


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an XML file containing the paragraph text from random Wikipedia pages.')
    parser.add_argument('--language', type=str, default='en', help='The language in which results will be returned.')
    parser.add_argument('--out', type=str, required=True, help='The output file.')
    parser.add_argument('--samples', type=int, default=10000, help='The number of Wikipedia pages to sample.')
    parser.add_argument('--threshold', type=int, default=200, help='The minimum number of characters required for a paragraph.')
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("The number of samples must be a positive integer.")

    if args.language not in wikipedia.languages():
        raise ValueError("Wikipedia version '" + args.language + "' cannot be found.")
    
    wikipedia.set_lang(args.language)
    wikipedia.set_rate_limiting(True)

    titles = set()
    root = ET.Element("texts")
    tree = ET.ElementTree(root)
    
    for i in range(args.samples):
        title = wikipedia.random()

        while True:
            try:
                title = wikipedia.random()

                while title in titles:
                    title = wikipedia.random()

                page = wikipedia.WikipediaPage(title=title)

                break

            except wikipedia.DisambiguationError as e:
                continue

            except requests.exceptions.ConnectionError:
                sleep(5)
                continue
        
        titles.add(title)

        print(i, title)
        content = page.content.split('\n')
        content = [x for x in content if x and not x.startswith('==') and not x.endswith('==') and len(x) > args.threshold]

        if content:
            paragraph = random.choice(content)
            txt = ET.SubElement(root, 'text')
            txt.text = paragraph
        else:
            continue
    
    with open(args.out, 'w') as f:
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
        f.write(xmlstr.encode('utf-8'))
