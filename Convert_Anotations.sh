#Beispiel wenn man zwei anotierte Datensätze kombinieren möchte, aber die Anotationen nicht ueberein stimmen.. zum ausführen sudo chmod +x Convert_Anotations.sh nicht vergessen
#!/bin/bash
for file in annotations/*.txt; do
    sed -i -E 's/^0 /2 /; s/^1 /0 /; s/^2 /1 /' "$file"
done
