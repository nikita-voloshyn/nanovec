# Scenariusz testowy Phase 2.5 — test interaktywny po polsku

## Cel

Ręcznie zweryfikować nowe narzędzia Phase 2.5 (`clear`, rozszerzony `stats`,
pola `distance` + `similarity`) na realnym, polskojęzycznym korpusie. Przy
okazji potwierdzić udokumentowane ograniczenie modelu MiniLM-L6-v2: gorszą
jakość rankingu dla treści innych niż angielska.

Test komplementuje zautomatyzowany `tests/integration/mcp_phase25.rs`, który
sprawdza tę samą powierzchnię, ale na sztucznych wektorach 384-wymiarowych.
Tutaj korzystamy z prawdziwych zapytań w języku polskim — żeby zobaczyć, jak
system zachowuje się w warunkach zbliżonych do produkcji.

---

## Wymagania wstępne

1. Gałąź zawiera commit `ed47f85` lub nowszy (Phase 2.5).
2. Bieżący `target/release/nanovec` zbudowany z tej gałęzi:
   ```bash
   cargo build --release
   ```
3. Klient MCP (np. Claude Code) **przeładowany** po podmianie binarki — bez
   tego serwer nadal serwuje starą poprzednią wersję bez `clear` i ze starym
   kształtem `stats`. W Claude Code: `/mcp` → odłączyć i podłączyć `nanovec`,
   albo zrestartować klienta.
4. Cache HuggingFace (`~/.cache/huggingface/hub/models--sentence-transformers
   --all-MiniLM-L6-v2/`) zainicjalizowany — pierwszy `embed` po
   reloadzie zajmie 10–30 s, kolejne ~30 ms.

---

## Mapa kroków

| # | Akcja | Oczekiwany wynik |
|---|-------|------------------|
| 1 | `stats` na pustej bazie | Pełny kształt z polami `default_metric` i `embedder` |
| 2 | Indeksowanie 7 dokumentów po polsku | ID 0–6, `stats.count == 7` |
| 3 | Zapytanie po polsku | Top-1 trafia w temat, ale gap do runner-up jest mały |
| 4 | Te same zapytania po angielsku | Wyraźnie czystszy ranking — kontrast z krokiem 3 |
| 5 | Weryfikacja wzorów `similarity` | `cosine: similarity = 1 − distance`, alias `score == distance` |
| 6 | `clear` i reset licznika ID | `{"deleted": 7}`, kolejne `index_*` daje `id: 0` |
| 7 | Idempotencja `clear` na pustej bazie | `{"deleted": 0}` bez błędu |

---

## Krok 1 — Stan początkowy

Po reloadzie klienta MCP wywołać:

```
stats()
```

Oczekiwana odpowiedź:

```json
{
  "count": 0,
  "dimension": 384,
  "default_metric": {
    "raw_vector": "euclidean",
    "document":   "cosine"
  },
  "embedder": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dim":   384
  },
  "metric": "Euclidean"
}
```

**Czerwona flaga:** brak `default_metric` lub `embedder` → klient MCP nadal
korzysta ze starej binarki sprzed Phase 2.5. Wrócić do punktu 3 wymagań
wstępnych.

---

## Krok 2 — Indeksowanie korpusu polskiego

Wywołać `index_document` siedem razy. Korpus jest podzielony na cztery
tematy (kalendarz, zakupy, incydenty produkcyjne, zdrowie), żeby wyszukiwarka
miała wystarczającą różnorodność semantyczną.

```text
1. index_document(
     text="Spotkanie zespołu inżynierskiego w piątek o 15:00 w sali konferencyjnej.",
     metadata={"category": "calendar"})

2. index_document(
     text="Przegląd kwartalnych wyników z dyrektorem we wtorek rano.",
     metadata={"category": "calendar"})

3. index_document(
     text="Kupić mleko, chleb i jajka po drodze do domu.",
     metadata={"category": "shopping"})

4. index_document(
     text="Zamówić nowy laptop dla zespołu projektowego.",
     metadata={"category": "shopping"})

5. index_document(
     text="Commit a3f9b2 zepsuł pipeline produkcyjny po południu.",
     metadata={"category": "incident"})

6. index_document(
     text="Baza danych zwraca timeout na zapytaniach analitycznych.",
     metadata={"category": "incident"})

7. index_document(
     text="Wizyta u dentysty zarezerwowana na poniedziałek po południu.",
     metadata={"category": "health"})
```

Każde wywołanie powinno zwrócić `{"id": N}` z `N` rosnącym od 0 do 6.

Sprawdzenie:

```
stats()    →    "count": 7
```

---

## Krok 3 — Wyszukiwanie po polsku

```
search_document(
  query="jakie spotkania mam zaplanowane w tym tygodniu?",
  k=5)
```

**Oczekiwane jakościowo:**

- Top-1: jeden z dokumentów `category: calendar` (id 0 albo id 1)
- `distance` w okolicy `0.6–0.9` (dla cosine na L2-normalized wektorach)
- `similarity = 1 − distance` w okolicy `0.1–0.4`
- **Gap** między top-1 a top-2: niewielki, często **< 0.10**

Powtórzyć dla pozostałych tematów:

```
search_document(query="co muszę kupić w sklepie?", k=5)
search_document(query="jakie są problemy z systemem produkcyjnym?", k=5)
search_document(query="kiedy mam wizytę u lekarza?", k=5)
```

**Co zapisać:**

| Zapytanie | Top-1 id | Top-1 distance | Top-1 similarity | Gap do top-2 |
|-----------|----------|----------------|------------------|--------------|
| spotkania | ?        | ?              | ?                | ?            |
| zakupy    | ?        | ?              | ?                | ?            |
| problemy  | ?        | ?              | ?                | ?            |
| lekarz    | ?        | ?              | ?                | ?            |

Mały gap to oczekiwany objaw słabości MiniLM-L6-v2 dla treści innych niż
angielska — udokumentowane w `docs/components/embed.md` w sekcji
"Model Limitations".

---

## Krok 4 — Te same zapytania po angielsku (kontrola)

```
search_document(query="what meetings do I have this week?", k=5)
search_document(query="what should I buy at the store?", k=5)
search_document(query="what production problems are there?", k=5)
search_document(query="when is my doctor appointment?", k=5)
```

**Co porównać z krokiem 3:**

- Distance dla top-1 powinien być wyraźnie mniejszy (~0.4–0.6 zamiast 0.6–0.9)
- Gap do top-2 powinien być wyraźnie większy (> 0.15)
- Top-1 powinien zawsze być z właściwej kategorii

Jeżeli różnica jest oczywista — to praktyczna demonstracja udokumentowanego
ograniczenia. Jeżeli nie ma różnicy — coś jest nie tak z embedderem lub
korpusem; sprawdzić logi serwera (stderr).

---

## Krok 5 — Weryfikacja wzorów `similarity`

Wziąć dowolny wynik z kroków 3 lub 4 i sprawdzić ręcznie:

| Pole | Definicja | Sprawdzenie |
|------|-----------|-------------|
| `score` | alias dla `distance` | `score == distance` (dokładnie) |
| `similarity` (cosine, default) | `1 − distance` | różnica < 1e-5 |

Wymusić pozostałe metryki na tym samym zapytaniu:

```
search_document(query="...", k=1, metric="euclidean")
# similarity = 1 / (1 + distance)

search_document(query="...", k=1, metric="dot")
# similarity = -distance  (= raw dot product, niegraniczony)
```

Ręczna weryfikacja:

- Dla cosine zapytanie identyczne z dokumentem dałoby `distance ≈ 0` i
  `similarity ≈ 1.0` — nie da się tego sprawdzić bezpośrednio dla zapytań
  naturalnych, ale jest to gwarantowane testem `phase_2_5_full_surface`.
- Dla euclidean: similarity zawsze w `(0, 1]`.
- Dla dot: similarity może być ujemny, jeśli wektory są przeciwlegle
  zorientowane — to nie błąd.

---

## Krok 6 — `clear` i reset licznika ID

```
clear()
```

Oczekiwane: `{"deleted": 7}`.

```
stats()
```

Oczekiwane:

- `count: 0`
- `dimension: 384` (zachowane)
- `embedder.model: "sentence-transformers/all-MiniLM-L6-v2"` (zachowane)

```
index_document(text="Pierwsze zdanie po wyczyszczeniu bazy.")
```

Oczekiwane: `{"id": 0}` — licznik ID został zresetowany przez `clear`.

```
search_document(query="jaka jest pierwsza fraza?", k=1)
```

Oczekiwane: jeden wynik, `id: 0`, distance niski (zapytanie blisko semantycznie).

---

## Krok 7 — Idempotencja `clear` na pustej bazie

Po kroku 6 baza ma jeden dokument. Wywołać:

```
clear()    →    {"deleted": 1}
clear()    →    {"deleted": 0}
```

Druga próba czyszczenia pustej bazy musi zwrócić `0` bez błędu. To gwarancja,
że klient może bezpiecznie wywołać `clear` przed nową sesją bez sprawdzania
`stats` w pętli.

---

## Kryteria akceptacji

Test zaliczony, jeżeli wszystkie poniższe są prawdziwe:

1. **Kształt `stats`**: zawiera `count`, `dimension`, `default_metric.raw_vector`,
   `default_metric.document`, `embedder.model`, `embedder.dim`, oraz alias
   `metric` na najwyższym poziomie.
2. **`clear` na niepustej bazie** zwraca dokładnie liczbę dokumentów przed
   czyszczeniem.
3. **`clear` na pustej bazie** zwraca `{"deleted": 0}` bez błędu.
4. **Reset ID**: po `clear` następne `index_*` zwraca `id: 0`.
5. **Pola wyniku**: każdy wynik `search` i `search_document` zawiera
   `id`, `text`, `metadata`, `distance`, `similarity`, `score`.
6. **Alias `score`**: dla każdego wyniku `score == distance`.
7. **Wzór cosine**: `similarity == 1.0 − distance` (tolerancja 1e-5).
8. **Wzór euclidean** (po wymuszeniu `metric: "euclidean"`):
   `similarity == 1.0 / (1.0 + distance)` (tolerancja 1e-5).
9. **Lock dimension**: `dimension == 384` przez cały czas trwania testu, w tym
   po `clear`.
10. **Kontrast PL vs EN**: zapytania angielskie z kroku 4 dają wyraźniejszy gap
    do runner-up niż polskie z kroku 3 (potwierdzenie znanej słabości modelu).

---

## Co nie jest w zakresie

- Pomiar wydajności (czas KNN, throughput indeksowania) → dla `/bench`.
- Testy SIMD → Phase 3.
- Wymiana modelu na multilingual → przyszła faza.
- Reżim współbieżny (wiele równoległych `index_document`) → osobny scenariusz.

---

## Diagnostyka i pułapki

| Symptom | Prawdopodobna przyczyna | Co zrobić |
|---------|-------------------------|-----------|
| Brak narzędzia `clear` na liście MCP | Klient nie przeładował binarki | `/mcp` → reconnect `nanovec` |
| `stats` zwraca tylko `count`, `dimension`, `metric` | Klient na starej wersji | jak wyżej |
| Kompletnie chaotyczny ranking po polsku | Znana słabość MiniLM | przeczytać `docs/components/embed.md`, sekcja "Model Limitations" |
| `search_document` zwraca `[]` mimo zindeksowanych dokumentów | Wymiar zapytania ≠ 384, albo embedder się nie załadował | sprawdzić stderr serwera, `embedder.dim` w `stats` |
| `score` ma wartość `null` lub brak go w odpowiedzi | Bug w aliasie backward-compat | otworzyć issue, podać dokładny request/response |

---

## Powiązane

- Plan: [`plans/phase-2.5-ergonomics-plan.md`](plans/phase-2.5-ergonomics-plan.md)
- Dispatch: [`plans/phase-2.5-ergonomics-dispatch.md`](plans/phase-2.5-ergonomics-dispatch.md)
- Raport: [`plans/phase-2.5-ergonomics-report.md`](plans/phase-2.5-ergonomics-report.md)
- Test automatyczny: `tests/integration/mcp_phase25.rs`
- Ograniczenia modelu: [`components/embed.md`](components/embed.md), sekcja "Model Limitations"
- Definicja pól wyniku: [`components/mcp-server.md`](components/mcp-server.md), sekcja "Score Fields (Phase 2.5)"
