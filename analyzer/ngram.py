from lxml import etree


class NGramAnalyzer:

    def __init__(self):
        with open("../resources/characters.txt") as f:
            self.characters = f.read() + " "

        self.characters_count = {}
        for c1 in self.characters:
            for c2 in self.characters:
                self.characters_count[c1 + c2] = 0
                self.total = 0
                self.totalx = 0

    def parse_xml_file(self, resource):
        print("Parsing " + resource + "...")
        tree = etree.parse(resource)
        root = tree.getroot()
        self.parse_xml_root(root)

    def parse_xml_root(self, root):
        title = root.tag.title()
        if title == "Text":
            self.parse_text(root.text)
        else:
            for children in root.getchildren():
                self.parse_xml_root(children)

    def parse_text(self, text):
        for i in range(1, len(text)):
            c1 = text[i - 1]
            c2 = text[i]
            if c1 not in self.characters or c2 not in self.characters:
                continue
            self.characters_count[c1 + c2] += 1
            if self.characters_count[c1 + c2] == 1:
                self.totalx += 1
            self.total += 1

    def get_stats(self):
        lst = []

        for key in self.characters_count:
            lst.append((key, self.characters_count[key]))

        lst.sort(key=lambda x: x[1], reverse=False)

        with open("result.txt","w") as f:
            cumul = 0
            j = 0
            for i in range(0, len(lst)):
                if lst[i][1] == 0:
                    continue
                cumul += lst[i][1]
                j = j + 1
                if i % 8 == 0:
                    f.write("(%.4f,%.4f)" % (j / self.totalx, cumul / self.total))


if __name__ == '__main__':
    analyser = NGramAnalyzer()
    analyser.parse_xml_file("../resources/texts/fr.xml.gz")
    analyser.parse_xml_file("../resources/texts/en.xml.gz")
    print(analyser.get_stats())
