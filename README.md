# Projekt z przedmiotu: Uczenie Maszynowe
Realizowane w zespole: Jan Jędrzejewski, Antoni Kowalczuk

## Temat projektu
Połączenie lasu losowego z SVM w zadaniu klasyfikacji. Postępujemy tak jak przy tworzeniu
lasu losowego, tylko co drugi klasyfikator w lesie to SVM. Jeden z klasyfikatorów (SVM lub
drzewo ID3) może pochodzić z istniejącej implementacji.

## Konfiguracja

Należy wywołać poniższe komendy
```
> python3 -m venv uma_env
> source uma_env/bin/activate
(uma_env) > python3 -m pip install -r requirements.txt
> ipython kernel install --user --name=uma_env
```

# Zaimplemetowane algorytmy

## ID3
Algorytm drzewa decyzyjnego

## Random Forest
Las losowy złożony z klasyfikatorów SVM i ID3. Jako hiperparametry przyjmuje:
- ilość klasyfikatorów w lesie
- hiperparametry klasyfikatorów bazowych
- ilość próbek w zbiorze treningowym każdeko z klasyfikatorów

Każdy z klasyfikatorów dokonuje predykcji dla pojedynczej próbki danych, następnie predykcje są agregowane i dostajemy predykcję lasu (kalsę) zgodnie ze strategią "najczęstszy=najlepszy".

# Wyniki
Szczegółowe rezultaty, porównanie działania, dobór parametrów oraz wnioski przedstawione są w dokumentacji końcowej projektu.