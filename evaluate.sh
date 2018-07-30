#!/bin/sh
cd results
ls *.txt > txt.lst
ls *.xml > xml.lst
java -jar /home/tbelos/git/structured-ocr/TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar xml.lst txt.lst
