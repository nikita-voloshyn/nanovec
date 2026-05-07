# Scenariusz testowy Phase 2.5 — wielojęzyczny korpus ScootGo

## Cel

Ręcznie zweryfikować nowe narzędzia Phase 2.5 (`clear`, rozszerzony `stats`,
pola `distance` + `similarity`) na realnym, **wielojęzycznym** korpusie
[`TEST-CORPUS-MULTILINGUAL.md`](TEST-CORPUS-MULTILINGUAL.md). Korpus zawiera
fikcyjne FAQ usługi wynajmu hulajnóg ScootGo — 10 tematów × 3 języki
(angielski, polski, ukraiński) = 30 chunków po jednym akapicie każdy.

Test idzie dalej niż automatyczny `tests/integration/mcp_phase25.rs` — sprawdza
**zachowanie modelu osadzającego MiniLM-L6-v2 przy zapytaniach w trzech
językach**, a przy okazji potwierdza, że wszystkie nowe pola Phase 2.5 są
spójnie obecne w odpowiedziach `search_document` na realnym ruchu.

---

## Wymagania wstępne

1. Gałąź zawiera commit `ed47f85` lub nowszy (Phase 2.5).
2. Bieżący `target/release/nanovec` zbudowany z tej gałęzi:
   ```bash
   cargo build --release
   ```
3. **Klient MCP przeładowany** po podmianie binarki — bez tego `clear` nie
   pojawi się w liście narzędzi, a `stats` nadal będzie zwracać stary kształt.
4. Cache HuggingFace zainicjalizowany (pierwszy `embed` po reloadzie zajmie
   10–30 s, kolejne ~30 ms).

Dokumenty pomocnicze pod ręką:

- [`TEST-CORPUS-MULTILINGUAL.md`](TEST-CORPUS-MULTILINGUAL.md) — korpus do indeksowania
- [`components/embed.md`](components/embed.md), sekcja "Model Limitations" — kontekst dla wyników
- [`components/mcp-server.md`](components/mcp-server.md), sekcja "Score Fields (Phase 2.5)" — definicja pól wyniku

---

## Mapa kroków

| # | Akcja | Co weryfikujemy |
|---|-------|-----------------|
| 1 | `stats` na pustej bazie | Nowy kształt z `default_metric` i `embedder` |
| 2 | Indeksowanie 30 chunków z korpusu | ID 0–29, `count: 30` |
| 3 | Same-language retrieval — PL→PL, EN→EN, UA→UA | Top-1 to chunk z **właściwego tematu** |
| 4 | Cross-lingual retrieval — jedno zapytanie, wszystkie języki | W top-3 powinny być wersje tego samego tematu (lub jednego języka) |
| 5 | Analiza gap'ów PL vs EN vs UA | EN ma największy gap, PL/UA wyraźnie mniejszy |
| 6 | Wzory `similarity` na realnych wynikach | `score == distance`, `similarity = 1 − distance` (cosine) |
| 7 | `clear` i reset licznika ID | `{"deleted": 30}`, kolejny `index_*` → `id: 0` |

---

## Krok 1 — Stan początkowy

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

**Czerwona flaga:** brak `default_metric` lub `embedder` → klient MCP nie
przeładował binarki sprzed Phase 2.5. Wrócić do punktu 3 wymagań wstępnych.

---

## Krok 2 — Indeksowanie korpusu

Korpus ma **stałą strukturę 30 chunków**: 10 tematów × 3 języki. Każdy chunk
indeksujemy oddzielnie z metadanymi `topic` (numer 1–10) oraz `lang` (`en` /
`pl` / `uk`). To daje łatwy filtr przy weryfikacji wyników.

**Schemat metadanych:**

```json
{
  "topic": "<1..10>",
  "lang":  "<en|pl|uk>",
  "section": "<krótka etykieta tematu>"
}
```

**Wzór wywołania** (powtórzyć dla wszystkich 30 chunków z `TEST-CORPUS-MULTILINGUAL.md`):

```text
index_document(
  text="To unlock a scooter, open the ScootGo app, scan the QR code...",
  metadata={"topic": "1", "lang": "en", "section": "unlocking"})

index_document(
  text="Aby odblokować hulajnogę, otwórz aplikację ScootGo, zeskanuj kod QR...",
  metadata={"topic": "1", "lang": "pl", "section": "unlocking"})

index_document(
  text="Щоб розблокувати самокат, відкрийте додаток ScootGo, відскануйте QR-код...",
  metadata={"topic": "1", "lang": "uk", "section": "unlocking"})

# … i tak dalej dla tematów 2–10
```

**Etykiety `section`** (skróty tematów dla łatwiejszej oceny wyników):

| topic | section |
|-------|---------|
| 1 | unlocking |
| 2 | payment |
| 3 | speed |
| 4 | helmet |
| 5 | parking |
| 6 | battery |
| 7 | damage |
| 8 | support |
| 9 | refunds |
| 10 | safety |

Po wszystkim:

```
stats()    →    "count": 30
```

Wszystkie zwrócone `id` powinny mieścić się w zakresie 0–29 i być rosnące.

---

## Krok 3 — Same-language retrieval

Dla każdego z trzech języków robimy zapytanie o **temat 5 (parkowanie)** w
tym samym języku. Spodziewany wynik: top-1 to chunk z `section: parking` w
języku zapytania.

```
search_document(query="where can I park the scooter?",         k=3)
search_document(query="gdzie mogę zaparkować hulajnogę?",       k=3)
search_document(query="де можна припаркувати самокат?",          k=3)
```

**Co zapisać** (osobno dla każdego zapytania):

| Zapytanie | Top-1 `section` | Top-1 `lang` | distance | similarity | Czy temat trafiony? | Czy język trafiony? |
|-----------|------------------|--------------|----------|------------|---------------------|---------------------|
| EN about parking | ? | ? | ? | ? | tak/nie | tak/nie |
| PL about parking | ? | ? | ? | ? | tak/nie | tak/nie |
| UA about parking | ? | ? | ? | ? | tak/nie | tak/nie |

**Hipoteza, którą testujemy:**

- Dla EN top-1 powinien być `{section: parking, lang: en}` z niskim distance
- Dla PL top-1 może być `{section: parking, lang: pl}` **albo**
  `{section: parking, lang: en}` (model lubi ciągnąć do angielskiego)
- Dla UA podobnie — możliwe wyciąganie EN-wersji w wyniku słabości
  multilingual w MiniLM

Powtórzyć dla **tematu 9 (zwroty pieniędzy)**:

```
search_document(query="can I get a refund?",                    k=3)
search_document(query="czy mogę otrzymać zwrot pieniędzy?",      k=3)
search_document(query="чи можна отримати повернення коштів?",    k=3)
```

I dla **tematu 1 (odblokowanie)**:

```
search_document(query="how do I unlock a scooter?",             k=3)
search_document(query="jak odblokować hulajnogę?",              k=3)
search_document(query="як розблокувати самокат?",                k=3)
```

---

## Krok 4 — Cross-lingual retrieval

Dla zapytania w jednym języku sprawdzamy, **czy wersje tego samego tematu w
innych językach też są w top-3**.

```
search_document(query="gdzie mogę zaparkować hulajnogę?", k=5)
```

Top-5 powinien zawierać 3 chunki z `section: parking` (po jednym na język),
ewentualnie zmieszane z bliskimi tematycznie (np. `topic 3 — speed`,
`topic 5 — parking` to powiązane reguły drogowe).

**Co weryfikujemy:**

| Pozycja | section | lang | distance |
|---------|---------|------|----------|
| 1 | parking | ? | ? |
| 2 | parking | ? | ? |
| 3 | parking | ? | ? |
| 4 | (inny temat) | ? | ? |
| 5 | (inny temat) | ? | ? |

Jeżeli top-3 to wszystkie trzy wersje językowe `section: parking`, model
robi przyzwoity cross-lingual matching dla tego tematu. Jeżeli top-3 to
mieszanka tematów po angielsku — model bardziej "lubi" angielski niż
trzyma się tematu.

---

## Krok 5 — Analiza gap'ów PL vs EN vs UA

Z trzech zapytań w kroku 3 (same temat — parkowanie) policzyć **gap między
top-1 a top-2** dla każdego języka:

```
gap = top2.distance - top1.distance
```

| Język | gap | Komentarz |
|-------|-----|-----------|
| EN | ? | spodziewane: największy gap, np. > 0.15 |
| PL | ? | spodziewane: mniejszy, np. 0.05–0.15 |
| UA | ? | spodziewane: najmniejszy lub porównywalny z PL |

**Interpretacja:**

- Duży gap = model jest pewny swojego top-1 (czyste trafienie)
- Mały gap = top-1 i top-2 prawie nieodróżnialne (top-1 może być przypadkowy)
- Jeżeli EN zdecydowanie wygrywa pod względem gap'u — to praktyczne
  potwierdzenie udokumentowanego ograniczenia MiniLM-L6-v2 z
  `docs/components/embed.md`.

---

## Krok 6 — Weryfikacja wzorów `similarity`

Z dowolnego wyniku z kroków 3–4 sprawdzić ręcznie:

| Pole | Definicja | Sprawdzenie |
|------|-----------|-------------|
| `score` | alias dla `distance` | `score == distance` (dokładnie) |
| `similarity` (cosine, default) | `1 − distance` | różnica < 1e-5 |

Wymusić pozostałe metryki na **tym samym zapytaniu**:

```
search_document(query="jak odblokować hulajnogę?", k=1, metric="euclidean")
# similarity = 1 / (1 + distance)

search_document(query="jak odblokować hulajnogę?", k=1, metric="dot")
# similarity = -distance  (= raw dot product)
```

Sprawdzić, że dla każdej metryki wzór jest zachowany.

---

## Krok 7 — `clear` i reset licznika ID

```
clear()
```

Oczekiwane: `{"deleted": 30}` — dokładnie liczba zindeksowanych chunków.

```
stats()
```

Oczekiwane:

- `count: 0`
- `dimension: 384` (zachowane)
- `embedder.model: "sentence-transformers/all-MiniLM-L6-v2"` (zachowane)
- `default_metric` — bez zmian

```
index_document(text="Pierwsze zdanie po wyczyszczeniu bazy.")
```

Oczekiwane: `{"id": 0}` — licznik ID zresetowany przez `clear`.

```
clear()
```

Oczekiwane: `{"deleted": 1}`.

```
clear()
```

Oczekiwane: `{"deleted": 0}` — idempotencja na pustej bazie.

---

## Kryteria akceptacji

Test zaliczony, jeżeli wszystkie poniższe są prawdziwe:

1. **Kształt `stats`**: zawiera `count`, `dimension`, `default_metric.raw_vector`,
   `default_metric.document`, `embedder.model`, `embedder.dim`, oraz alias
   `metric` na najwyższym poziomie.
2. **Indeksowanie korpusu**: wszystkie 30 wywołań `index_document` zwracają
   rosnące `id` w zakresie 0–29. Po wszystkim `stats.count == 30`.
3. **Same-language top-1 dla EN**: zapytania angielskie z kroku 3 zawsze
   trafiają top-1 z prawidłową `section` (model jest trenowany głównie po
   angielsku).
4. **Same-language top-1 dla PL/UA — best effort**: w idealnym przypadku
   trafiamy `section`, ale ze względu na ograniczenie multilingual MiniLM
   pojedyncze pomyłki tematu są dopuszczalne. Brak trafień w temacie w
   więcej niż 1 z 3 zapytań → potwierdzenie znanej słabości modelu, **nie**
   regresja Phase 2.5.
5. **Pola wyniku**: każdy wynik `search_document` zawiera `id`, `text`,
   `metadata`, `distance`, `similarity`, `score`.
6. **Alias `score`**: dla każdego wyniku `score == distance`.
7. **Wzór cosine**: `similarity == 1.0 − distance` (tolerancja 1e-5).
8. **Wzór euclidean**: `similarity == 1.0 / (1.0 + distance)` (tolerancja 1e-5).
9. **Wzór dot**: `similarity == -distance`.
10. **Reset ID po `clear`**: następne `index_*` zwraca `id: 0`.
11. **Idempotencja `clear`**: drugie wywołanie na pustej bazie zwraca
    `{"deleted": 0}` bez błędu.
12. **Lock dimension**: `dimension == 384` przez cały czas trwania testu.

---

## Co nie jest w zakresie

- Pomiar wydajności (czas KNN, throughput indeksowania) → dla `/bench`.
- Testy SIMD → Phase 3.
- Wymiana modelu na multilingual e5 / bge-m3 → przyszła faza.
- Reżim współbieżny (wiele równoległych `index_document`) → osobny scenariusz.

---

## Diagnostyka i pułapki

| Symptom | Prawdopodobna przyczyna | Co zrobić |
|---------|-------------------------|-----------|
| Brak narzędzia `clear` na liście MCP | Klient nie przeładował binarki | `/mcp` → reconnect `nanovec`, sprawdzić czy `target/release/nanovec` jest świeży |
| `stats` zwraca tylko `count`, `dimension`, `metric` | Klient na starej wersji | jak wyżej |
| `search_document` zwraca `[]` dla niepustej bazy | Embedder nie załadował się prawidłowo | sprawdzić stderr serwera (logowanie tracing), `embedder.dim` w `stats` |
| Top-1 dla PL/UA z innego tematu | Znana słabość MiniLM-L6-v2 dla treści innych niż angielska | nie regresja — porównać z EN, jeżeli EN czyste, to model ma trudność z polskim/ukraińskim |
| `score` == 0 dla wszystkich wyników | Bug w aliasie backward-compat | otworzyć issue, dołączyć dokładny request/response |
| `similarity` nie zgadza się z formułą | Bug w `similarity_for` | sprawdzić użytą metrykę (`metric` w wywołaniu, `default_metric` w `stats`), bo formuła jest metryko-zależna |
| Embedder ładuje się 30+ s przy każdym restarcie | Brak cache HuggingFace | sprawdzić `~/.cache/huggingface/hub/`, ewentualnie zaakceptować jako koszt jednorazowy |

---

## Powiązane

- Korpus testowy: [`TEST-CORPUS-MULTILINGUAL.md`](TEST-CORPUS-MULTILINGUAL.md)
- Plan: [`plans/phase-2.5-ergonomics-plan.md`](plans/phase-2.5-ergonomics-plan.md)
- Dispatch: [`plans/phase-2.5-ergonomics-dispatch.md`](plans/phase-2.5-ergonomics-dispatch.md)
- Raport: [`plans/phase-2.5-ergonomics-report.md`](plans/phase-2.5-ergonomics-report.md)
- Test automatyczny: `tests/integration/mcp_phase25.rs`
- Ograniczenia modelu: [`components/embed.md`](components/embed.md), sekcja "Model Limitations"
- Definicja pól wyniku: [`components/mcp-server.md`](components/mcp-server.md), sekcja "Score Fields (Phase 2.5)"
