"""Generator eval set v4 — CERMIN 8 kategori resmi Track 1 (gaya practice tasks).

Beda dari v3 (math-heavy, utk kalibrasi arsitektur lama): v4 menyeimbangkan
kedelapan kategori resmi dan memberi tiap task grader yang paling deterministik
yang mungkin:
  - exact         : salah satu string jawaban muncul (boundary-aware)
  - contains_all  : SEMUA string wajib muncul (NER, factual multi-part)
  - pytests       : ekstrak kode dari jawaban, jalankan + assert (code gen/debug)
  - judge         : LLM independen (gemma3:12b) vs reference (summarisation)

Jalankan:  python -m eval.gen_tasks_v4  ->  eval/tasks_v4.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "tasks_v4.jsonl"

TASKS: list[dict] = []


def add(category: str, prompt: str, grader: str, *, answer=None, reference=None, tests=None):
    t = {"id": len(TASKS) + 1, "category": category, "input": prompt, "grader": grader}
    if answer is not None:
        t["answer"] = answer if isinstance(answer, list) else [str(answer)]
    if reference is not None:
        t["reference"] = reference
    if tests is not None:
        t["tests"] = tests
    TASKS.append(t)


# ---------------- factual ----------------
add("factual", "What is the capital of Australia, and what body of water is it near?",
    "contains_all", answer=["Canberra", "Burley Griffin"])
add("factual", "What is the chemical symbol for gold?", "exact", answer=["Au"])
add("factual", "Which planet in our solar system is known as the Red Planet?", "exact", answer=["Mars"])
add("factual", "Who wrote the novel Nineteen Eighty-Four?", "exact", answer=["Orwell"])
add("factual", "What is the largest ocean on Earth?", "exact", answer=["Pacific"])
add("factual", "In what year did the Berlin Wall fall?", "exact", answer=["1989"])
add("factual", "Which chemical element has atomic number 6?", "exact", answer=["carbon"])
add("factual", "What is the longest river in South America?", "exact", answer=["Amazon"])
add("factual", "What gas do plants primarily absorb from the air during photosynthesis?",
    "exact", answer=["carbon dioxide", "CO2"])
add("factual", "Which country hosted the first modern Olympic Games in 1896, and in which city?",
    "contains_all", answer=["Greece", "Athens"])

# ---------------- math reasoning ----------------
add("math", "A store has 320 items. It sells 25% of them on Monday and 40 more on Tuesday. "
    "How many items remain?", "exact", answer=["200"])
add("math", "A jacket costs $150. It is discounted 20%, then an additional 10% off the reduced "
    "price. What is the final price in dollars?", "exact", answer=["108"])
add("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
    "exact", answer=["72"])
add("math", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost in dollars?", "exact", answer=["0.05", "$0.05", ".05", "5 cents"])
add("math", "What is 17 multiplied by 24?", "exact", answer=["408"])
add("math", "What is the sum of all integers from 1 to 40?", "exact", answer=["820"])
add("math", "What is the greatest common divisor of 84 and 36?", "exact", answer=["12"])
add("math", "A snail climbs a 12-meter well: it climbs 4 meters each day and slips back 3 meters "
    "each night. On which day does it reach the top?", "exact", answer=["9", "ninth"])
add("math", "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take "
    "100 machines to make 100 widgets?", "exact", answer=["5", "five"])
add("math", "What is 15% of 240?", "exact", answer=["36"])

# ---------------- sentiment ----------------
add("sentiment", "Classify the sentiment of this review: The food was absolutely delicious and "
    "the staff were wonderful.", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this review: Terrible service, cold food, and we "
    "waited an hour. Never coming back.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this statement: The package arrived on the scheduled "
    "delivery date.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: The battery life is great, but the "
    "screen scratches far too easily.", "exact", answer=["mixed", "neutral"])
add("sentiment", "Classify the sentiment of this review: I was skeptical at first, but this "
    "vacuum exceeded every expectation. Worth every penny!", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this tweet: Flight delayed four hours, lost my "
    "luggage, and customer service hung up on me.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this sentence: The meeting has been moved from "
    "3 PM to 4 PM on Thursday.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: Gorgeous design and blazing fast, "
    "though the price stings a little.", "exact", answer=["positive", "mixed"])

# ---------------- summarisation (judge + constraint) ----------------
add("summarisation", "Summarize the following in exactly one sentence: The new library opened "
    "downtown last week. It features a rooftop garden, a children's wing, and a cafe. Residents "
    "have praised its modern design and free public workshops.", "judge",
    reference="The new downtown library, which opened last week with a rooftop garden, "
              "children's wing, and cafe, has been praised by residents for its modern design "
              "and free workshops.")
add("summarisation", "Summarize the following in exactly one sentence: Scientists discovered a "
    "new species of deep-sea fish near the Mariana Trench. The fish uses bioluminescence to "
    "attract prey. It can survive extreme pressure at depths of over 8,000 meters.", "judge",
    reference="Scientists discovered a new bioluminescent deep-sea fish near the Mariana Trench "
              "that survives at depths over 8,000 meters.")
add("summarisation", "Summarize the following in one sentence: The city council voted to expand "
    "the bike lane network by 40 kilometers. Construction begins in March and should finish by "
    "October. Local businesses expressed concerns about parking, but cycling groups celebrated "
    "the decision.", "judge",
    reference="The city council approved a 40 km bike lane expansion running March to October, "
              "welcomed by cyclists but worrying businesses about parking.")
add("summarisation", "Summarize the following in exactly two sentences: Remote work adoption "
    "surged during the pandemic and has remained high. Many companies now offer hybrid "
    "schedules. Studies show productivity stayed stable or improved, though some managers worry "
    "about collaboration and mentoring of junior staff.", "judge",
    reference="Remote work rose sharply during the pandemic and remains common, with many firms "
              "adopting hybrid schedules. Productivity has held steady or improved, although "
              "managers still worry about collaboration and mentoring.")
add("summarisation", "Summarize the following in one sentence: A regional airline announced it "
    "will retire its aging turboprop fleet next year. The planes will be replaced with newer, "
    "more fuel-efficient jets. The transition is expected to cut fuel costs by 30 percent and "
    "reduce noise complaints from residents near the airport.", "judge",
    reference="A regional airline will replace its old turboprops with fuel-efficient jets next "
              "year, cutting fuel costs 30% and reducing noise.")
add("summarisation", "Condense this into a single sentence: The museum's new exhibit features "
    "artifacts from ancient Mesopotamia, including clay tablets with cuneiform writing. Curators "
    "spent five years assembling the collection from loans by twelve institutions.", "judge",
    reference="The museum's new ancient-Mesopotamia exhibit, five years in the making with loans "
              "from twelve institutions, features artifacts including cuneiform clay tablets.")

# ---------------- NER ----------------
add("ner", "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI "
    "in Berlin last March.", "contains_all", answer=["Maria Sanchez", "Fireworks AI", "Berlin", "March"])
add("ner", "Extract all named entities and their types from: Tim Cook announced the partnership "
    "at Apple Park in Cupertino on Tuesday.", "contains_all",
    answer=["Tim Cook", "Apple Park", "Cupertino", "Tuesday"])
add("ner", "Extract the named entities and their types from: Dr. Amara Okafor presented her "
    "research at Oxford University in September.", "contains_all",
    answer=["Amara Okafor", "Oxford University", "September"])
add("ner", "Identify all named entities and their types in: Toyota opened a new plant in "
    "Guadalajara, Mexico, creating 3,000 jobs.", "contains_all",
    answer=["Toyota", "Guadalajara", "Mexico"])
add("ner", "Extract all named entities and their types from: The treaty was signed by Chancellor "
    "Weber and President Silva in Geneva on 12 June 2024.", "contains_all",
    answer=["Weber", "Silva", "Geneva", "12 June 2024"])
add("ner", "Extract named entities and their types from: Lionel Messi scored twice as Inter "
    "Miami beat Orlando City on Saturday.", "contains_all",
    answer=["Lionel Messi", "Inter Miami", "Orlando City", "Saturday"])
add("ner", "List all named entities with their types from: Samsung unveiled the Galaxy Fold at "
    "its headquarters in Seoul during January.", "contains_all",
    answer=["Samsung", "Galaxy Fold", "Seoul", "January"])
add("ner", "Extract all named entities and their types from: NASA launched the Artemis mission "
    "from Kennedy Space Center in Florida.", "contains_all",
    answer=["NASA", "Artemis", "Kennedy Space Center", "Florida"])

# ---------------- code debugging (pytests) ----------------
add("code_debug", "This function should return the max of a list but has a bug: "
    "def get_max(nums): return nums[0]. Find and fix it.", "pytests",
    tests="assert get_max([3,1,2])==3\nassert get_max([-5,-2,-9])==-2\nassert get_max([7])==7")
add("code_debug", "This function should sum the integers from 1 to n inclusive but has a bug: "
    "def sum_to_n(n): return sum(range(n)). Fix it.", "pytests",
    tests="assert sum_to_n(5)==15\nassert sum_to_n(1)==1\nassert sum_to_n(10)==55")
add("code_debug", "This function should return True when n is even, but it is wrong: "
    "def is_even(n): return n % 2 == 1. Fix it.", "pytests",
    tests="assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True")
add("code_debug", "This function should reverse a string but has a bug: "
    "def reverse(s): return s[::2]. Fix it.", "pytests",
    tests="assert reverse('abc')=='cba'\nassert reverse('hello')=='olleh'\nassert reverse('')==''")
add("code_debug", "This function should return the average as a float but has a bug: "
    "def average(nums): return sum(nums) // len(nums). Fix it.", "pytests",
    tests="assert average([1,2])==1.5\nassert average([3,3,3])==3.0")
add("code_debug", "This recursive factorial has a bug: "
    "def fact(n):\n    if n == 1: return 0\n    return n * fact(n-1)\nFix it.", "pytests",
    tests="assert fact(1)==1\nassert fact(5)==120\nassert fact(3)==6")
add("code_debug", "This function should count how many times x appears in a list but is buggy: "
    "def count_x(lst, x): return len(lst) - lst.count(x). Fix it.", "pytests",
    tests="assert count_x([1,2,2,3],2)==2\nassert count_x([],5)==0\nassert count_x([4,4,4],4)==3")
add("code_debug", "This function should return the first word of a sentence but has a bug: "
    "def first_word(s): return s.split()[-1]. Fix it.", "pytests",
    tests="assert first_word('hello world')=='hello'\nassert first_word('one')=='one'")

# ---------------- logical / deductive ----------------
add("logical", "Three friends, Ana, Ben, and Carl, each own a different pet: cat, dog, bird. "
    "Ana does not own the bird. Ben owns the dog. Who owns the cat?", "exact", answer=["Ana"])
add("logical", "All roses are flowers. Some flowers fade quickly. Can we logically conclude "
    "that some roses fade quickly? Answer yes or no.", "exact", answer=["no"])
add("logical", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally "
    "have?", "exact", answer=["1", "one"])
add("logical", "There are three killers in a room. Someone enters and kills one of them. Nobody "
    "leaves the room. How many killers are left in the room?", "exact", answer=["3", "three"])
add("logical", "Tom is taller than Jim. Jim is taller than Sue. Who is the shortest?",
    "exact", answer=["Sue"])
add("logical", "A farmer has 17 sheep. All but 9 run away. How many sheep are left?",
    "exact", answer=["9", "nine"])
add("logical", "If two days after tomorrow is Friday, what day is today?", "exact",
    answer=["Tuesday"])
add("logical", "On an island, knights always tell the truth and knaves always lie. A person "
    "says: 'I am a knave.' Is this possible? Answer yes or no.", "exact", answer=["no"])

# ---------------- code generation (pytests) ----------------
add("code_gen", "Write a Python function named second_largest(nums) that returns the "
    "second-largest distinct number in a list, handling duplicates correctly.", "pytests",
    tests="assert second_largest([1,2,3])==2\nassert second_largest([5,5,4])==4\n"
          "assert second_largest([10,10,10,7])==7")
add("code_gen", "Write a Python function named reverse_words(s) that reverses the order of "
    "words in a string. Words are separated by single spaces.", "pytests",
    tests="assert reverse_words('hello world')=='world hello'\n"
          "assert reverse_words('a b c')=='c b a'\nassert reverse_words('one')=='one'")
add("code_gen", "Write a Python function named is_palindrome(s) that returns True if the string "
    "is a palindrome, ignoring case.", "pytests",
    tests="assert is_palindrome('Racecar') is True\nassert is_palindrome('hello') is False\n"
          "assert is_palindrome('') is True")
add("code_gen", "Write a Python function named fizzbuzz(n) that returns 'Fizz' if n is divisible "
    "by 3, 'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, otherwise the number as a "
    "string.", "pytests",
    tests="assert fizzbuzz(3)=='Fizz'\nassert fizzbuzz(5)=='Buzz'\nassert fizzbuzz(15)=='FizzBuzz'\n"
          "assert fizzbuzz(7)=='7'")
add("code_gen", "Write a Python function named count_vowels(s) that returns the number of vowels "
    "(a, e, i, o, u, case-insensitive) in a string.", "pytests",
    tests="assert count_vowels('hello')==2\nassert count_vowels('AEIOU')==5\nassert count_vowels('xyz')==0")
add("code_gen", "Write a Python function named flatten(lst) that flattens a list of lists one "
    "level deep into a single list.", "pytests",
    tests="assert flatten([[1,2],[3],[4,5]])==[1,2,3,4,5]\nassert flatten([])==[]\n"
          "assert flatten([[],[1]])==[1]")
add("code_gen", "Write a Python function named nth_fib(n) that returns the nth Fibonacci number "
    "iteratively, where nth_fib(0)=0 and nth_fib(1)=1.", "pytests",
    tests="assert nth_fib(0)==0\nassert nth_fib(1)==1\nassert nth_fib(10)==55")
add("code_gen", "Write a Python function named are_anagrams(a, b) that returns True if the two "
    "strings are anagrams of each other, ignoring case.", "pytests",
    tests="assert are_anagrams('Listen','Silent') is True\nassert are_anagrams('cat','dog') is False\n"
          "assert are_anagrams('a','a') is True")


def main():
    with open(OUT, "w") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    from collections import Counter
    c = Counter(t["category"] for t in TASKS)
    g = Counter(t["grader"] for t in TASKS)
    print(f"Wrote {len(TASKS)} tasks -> {OUT}")
    print("  per category:", dict(c))
    print("  per grader  :", dict(g))


if __name__ == "__main__":
    main()
