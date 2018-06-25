#!/bin/sh
cd results
ls *.txt > txt.lst
ls *.xml > xml.lst
java -jar /space_sde/tbelos/git/TranskribusBaseLineEvaluationScheme/target/TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar xml.lst txt.lst

