import time
from collections import defaultdict

from models import Library
from models.solution import Solution
import copy
import random
from collections import deque


class Solver:
    def __init__(self):
        pass

    def build_grasp_solution(self, data, p=0.05):
        """
        Build a feasible solution using a GRASP-like approach:
          - Sorting libraries by signup_days ASC, then total_score DESC.
          - Repeatedly choosing from the top p% feasible libraries at random.

        Args:
            data: The problem data (libraries, scores, num_days, etc.)
            p: Percentage (as a fraction) for the restricted candidate list (RCL)

        Returns:
            A Solution object with the constructed solution
        """
        libs_sorted = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)),
        )

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        candidate_libs = libs_sorted[:]

        while candidate_libs:
            rcl_size = max(1, int(len(candidate_libs) * p))
            rcl = candidate_libs[:rcl_size]

            chosen_lib = random.choice(rcl)
            candidate_libs.remove(chosen_lib)

            if curr_time + chosen_lib.signup_days >= data.num_days:
                unsigned_libraries.append(chosen_lib.id)
            else:
                time_left = data.num_days - (curr_time + chosen_lib.signup_days)
                max_books_scanned = time_left * chosen_lib.books_per_day

                available_books = sorted(
                    {book.id for book in chosen_lib.books} - scanned_books,
                    key=lambda b: -data.scores[b],
                )[:max_books_scanned]

                if available_books:
                    signed_libraries.append(chosen_lib.id)
                    scanned_books_per_library[chosen_lib.id] = available_books
                    scanned_books.update(available_books)
                    curr_time += chosen_lib.signup_days
                else:
                    unsigned_libraries.append(chosen_lib.id)

        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        solution.calculate_fitness_score(data.scores)
        return solution

    def generate_initial_solution_grasp(self, data, p=0.05, max_time=60):
        """
        Generate an initial solution using a GRASP-like approach:
          1) Sort libraries by (signup_days ASC, total_score DESC).
          2) Repeatedly pick from top p% of feasible libraries at random.
          3) Optionally improve with a quick local search for up to max_time seconds.

        :param data:      The problem data (libraries, scores, num_days, etc.).
        :param p:         Percentage (as a fraction) for the restricted candidate list (RCL).
        :param max_time:  Time limit (in seconds) to repeat GRASP + local search.
        :return:          A Solution object with the best found solution.
        """
        start_time = time.time()
        best_solution = None
        Library._id_counter = 0

        while time.time() - start_time < max_time:
            candidate_solution = self.build_grasp_solution(data, p)

            improved_solution = self.local_search(
                candidate_solution, data, time_limit=1.0
            )

            if (best_solution is None) or (
                    improved_solution.fitness_score > best_solution.fitness_score
            ):
                best_solution = improved_solution

        return best_solution

    def local_search(self, solution, data, time_limit=1.0):
        """
        A simple local search/hill-climbing method that randomly selects one of the available tweak methods.
        Uses choose_tweak_method to select the tweak operation based on defined probabilities.
        Runs for 'time_limit' seconds and tries small random modifications.
        """
        start_time = time.time()
        best = copy.deepcopy(solution)

        while time.time() - start_time < time_limit:
            selected_tweak = self.choose_tweak_method()

            neighbor = selected_tweak(copy.deepcopy(best), data)
            if neighbor.fitness_score > best.fitness_score:
                best = neighbor

        return best

    def choose_tweak_method(self):
        """Randomly chooses a tweak method based on the defined probabilities."""
        tweak_methods = [
            (self.tweak_solution_swap_signed_with_unsigned, 0.5),
            (self.tweak_solution_swap_same_books, 0.1),
            (self.crossover, 0.2),
            (self.tweak_solution_swap_last_book, 0.1),
            (self.tweak_solution_swap_signed, 0.1),
        ]

        methods, weights = zip(*tweak_methods)

        selected_method = random.choices(methods, weights=weights, k=1)[0]
        return selected_method

    def crossover(self, solution, data):
        """Performs crossover by shuffling library order and swapping books accordingly."""
        new_solution = copy.deepcopy(solution)

        old_order = new_solution.signed_libraries[:]
        library_indices = list(range(len(data.libs)))
        random.shuffle(library_indices)

        new_scanned_books_per_library = {}

        for new_idx, new_lib_idx in enumerate(library_indices):
            if new_idx >= len(old_order):
                break

            old_lib_id = old_order[new_idx]
            new_lib_id = new_lib_idx

            if new_lib_id < 0 or new_lib_id >= len(data.libs):
                print(f"Warning: new_lib_id {new_lib_id} is out of range for data.libs (size: {len(data.libs)})")
                continue

            if old_lib_id in new_solution.scanned_books_per_library:
                books_to_move = new_solution.scanned_books_per_library[old_lib_id]

                existing_books_in_new_lib = {book.id for book in data.libs[new_lib_id].books}

                valid_books = []
                for book_id in books_to_move:
                    if book_id not in existing_books_in_new_lib and book_id not in [b for b in valid_books]:
                        valid_books.append(book_id)

                new_scanned_books_per_library[new_lib_id] = valid_books

        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_signed(self, solution, data):
        book_count = defaultdict(int)
        unscanned_books_per_library = {}

        for library in data.libs:
            if library.id in solution.signed_libraries:
                unsigned_books = []
                for book in library.books:
                    book_count[book.id] += 1
                    if book.id not in solution.scanned_books_per_library.get(library.id,
                                                                             []) and book.id not in solution.scanned_books:
                        unsigned_books.append(book.id)
                if len(unsigned_books) > 0:
                    unscanned_books_per_library[library.id] = unsigned_books

        if len(unscanned_books_per_library) == 1:
            # print("Only 1 library with unscanned books was found")
            return solution

        possible_books = [
            book_id for book_id, count in book_count.items()
            if count > 1 and book_id in solution.scanned_books
        ]

        valid_books = set()

        for library, unscanned_books in unscanned_books_per_library.items():
            for book_id in possible_books:
                for book in data.libs[library].books:
                    if book.id == book_id:
                        valid_books.add(book_id)

        if not valid_books:
            # print("No valid books were found")
            return solution  # No book meets the criteria, return unchanged

        # Get random book to swap
        book_to_move = random.choice(list(valid_books))

        # Identify which library is currently scanning this book
        current_library = None
        for lib_id, books in solution.scanned_books_per_library.items():
            if book_to_move in books:
                current_library = lib_id
                break

        if unscanned_books_per_library.get(current_library) is None or len(
                unscanned_books_per_library[current_library]) == 0:
            return solution

        # Select other library with any un-scanned books to scan this book
        possible_libraries = [
            lib for lib in data.book_libs[book_to_move]
            if lib != current_library and any(
                library.id == lib and any(book.id not in solution.scanned_books for book in library.books)
                for library in data.libs if library.id in solution.signed_libraries
            )
        ]

        if len(possible_libraries) == 0:
            # print("No valid libraries were found")
            return solution

        new_library = random.choice(possible_libraries)

        # Remove the book from the current library
        solution.scanned_books_per_library[current_library].remove(book_to_move)
        solution.scanned_books.remove(book_to_move)

        # Add the book to the new library, maintaining feasibility
        current_books_in_new_library = solution.scanned_books_per_library[new_library]

        # Ensure feasibility: If new_library is at its limit, remove a book to make space
        max_books_per_day = data.libs[new_library].books_per_day

        days_before_sign_up = 0
        found = False

        for id in solution.signed_libraries:
            if found:
                break
            days_before_sign_up += data.libs[id].signup_days
            if id == new_library:
                found = True

        numOfDaysAvailable = data.num_days - days_before_sign_up

        book_to_remove = None
        if len(current_books_in_new_library) > numOfDaysAvailable * max_books_per_day:
            book_to_remove = random.choice(list(current_books_in_new_library))
            current_books_in_new_library.remove(book_to_remove)
            solution.scanned_books.remove(book_to_remove)

        # Add the book to the new library
        current_books_in_new_library.append(book_to_move)
        solution.scanned_books.add(book_to_move)

        books_in_current_library = solution.scanned_books_per_library[current_library]

        new_scanned_book = random.choice(unscanned_books_per_library.get(current_library))
        books_in_current_library.append(new_scanned_book)
        solution.scanned_books.add(new_scanned_book)

        solution.calculate_delta_fitness(data, new_scanned_book, book_to_remove)

        return solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2 / 3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        local_signed_libs = solution.signed_libraries.copy()
        local_unsigned_libs = solution.unsigned_libraries.copy()

        total_signed = len(local_signed_libs)

        # Bias
        if bias_type == "favor_first_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(local_unsigned_libs) - 1)

        # signed_lib_id = self._extract_lib_id(local_signed_libs, signed_idx)
        # unsigned_lib_id = self._extract_lib_id(local_unsigned_libs, unsigned_idx)
        signed_lib_id = local_signed_libs[signed_idx]
        unsigned_lib_id = local_unsigned_libs[unsigned_idx]

        # Swap the libraries
        local_signed_libs[signed_idx] = unsigned_lib_id
        local_unsigned_libs[unsigned_idx] = signed_lib_id
        # print(f"swapped_signed_lib={unsigned_lib_id}")
        # print(f"swapped_unsigned_lib={unsigned_lib_id}")

        # Preserve the part before `signed_idx`
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        lib_lookup = {lib.id: lib for lib in data.libs}

        # Process libraries before the swapped index
        for i in range(signed_idx):
            # lib_id = self._extract_lib_id(solution.signed_libraries, i)
            lib_id = solution.signed_libraries[i]
            library = lib_lookup.get(lib_id)

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Recalculate from `signed_idx` onward
        new_signed_libraries = local_signed_libs[:signed_idx]

        for i in range(signed_idx, len(local_signed_libs)):
            # lib_id = self._extract_lib_id(local_signed_libs, i)
            lib_id = local_signed_libs[i]
            library = lib_lookup.get(lib_id)

            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(library.id)
                continue

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_signed_libraries.append(library.id)  # Not f"Library {library.id}"
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Update solution
        new_solution = Solution(new_signed_libraries, local_unsigned_libs, new_scanned_books_per_library, scanned_books)
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]

        if len(library_ids) < 2:
            return solution

        idx1 = random.randint(0, len(library_ids) - 1)
        idx2 = random.randint(0, len(library_ids) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(library_ids) - 1)

        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]

        ordered_libs = [data.libs[lib_id] for lib_id in library_ids]

        all_lib_ids = set(range(len(data.libs)))
        remaining_lib_ids = all_lib_ids - set(library_ids)
        for lib_id in sorted(remaining_lib_ids):
            ordered_libs.append(data.libs[lib_id])

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in ordered_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def hill_climbing_combined_w_initial_solution(self, solution, data, iterations=1000):

        list_of_climbs = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            # self.tweak_solution_swap_signed,
            self.tweak_solution_swap_last_book
        ]

        for i in range(iterations - 1):
            # if i % 100 == 0:
            #     print('i',i)
            target_climb = random.choice(list_of_climbs)
            # solution_copy = copy.deepcopy(solution)
            # new_solution = target_climb(solution_copy, data)

            # solution_copy = copy.deepcopy(solution)
            new_solution = target_climb(solution, data)

            if (new_solution.fitness_score >= solution.fitness_score):
                solution = new_solution

        return (solution.fitness_score, solution)

    def tweak_solution_swap_last_book(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution  # No scanned or unsigned libraries, return unchanged solution

        # Pick a random library that has scanned books
        # In solver.py, make sure to fix the first issue by copying correctly:
        local_signed_libs = solution.signed_libraries.copy()  # Assuming signed_libraries is a list or dictionary

        # In solver.py, for the second issue, use range(len()) instead of .keys():
        chosen_lib_id = random.choice(range(len(solution.scanned_books_per_library)))


        if not chosen_lib_id:
            return solution  # Safety check, shouldn't happen

        books = solution.scanned_books_per_library.get(chosen_lib_id, [])
    
        if books:
            last_scanned_book = books[-1]
            # Continue logic here using last_scanned_book...
        else:
            return solution  # or handle the case where the library has no scanned books
        # Get the last scanned book from this library

        library_dict = {lib.id: lib for lib in data.libs}

        best_book = None
        best_score = -1

        for unsigned_lib in solution.unsigned_libraries:
            library = library_dict[unsigned_lib]  # O(1) dictionary lookup

            # Find the first unscanned book from this library
            for book in library.books:
                if book.id not in solution.scanned_books:  # O(1) lookup in set
                    if data.scores[book.id] > best_score:  # Only store the best
                        best_book = book.id
                        best_score = data.scores[book.id]
                    break  # Stop after the first valid book

        # Assign the best book found (or None if none exist)
        first_unscanned_book = best_book

        if first_unscanned_book is None:
            return solution  # No available unscanned books

        # Create new scanned books mapping (deep copy)
        new_scanned_books_per_library = {
            lib_id: books.copy() for lib_id, books in solution.scanned_books_per_library.items()
        }

        # Swap the books
        new_scanned_books_per_library[chosen_lib_id].remove(last_scanned_book)
        new_scanned_books_per_library[chosen_lib_id].append(first_unscanned_book)

        # Update the overall scanned books set
        new_scanned_books = solution.scanned_books.copy()
        new_scanned_books.remove(last_scanned_book)
        new_scanned_books.add(first_unscanned_book)

        # Create the new solution
        new_solution = Solution(
            signed_libs=solution.signed_libraries.copy(),
            unsigned_libs=solution.unsigned_libraries.copy(),
            scanned_books_per_library=new_scanned_books_per_library,
            scanned_books=new_scanned_books
        )

        # Recalculate fitness score
        new_solution.calculate_fitness_score(data.scores)

        return new_solution
    
    def tweak_solution_swap_neighbor_libraries(self, solution, data):
        """Swaps two adjacent libraries in the signed list to create a neighbor solution."""
        if len(solution.signed_libraries) < 2:
            return solution
 
        new_solution = copy.deepcopy(solution)
        swap_pos = random.randint(0, len(new_solution.signed_libraries) - 2)
 
        # Swap adjacent libraries
        new_solution.signed_libraries[swap_pos], new_solution.signed_libraries[swap_pos + 1] = \
            new_solution.signed_libraries[swap_pos + 1], new_solution.signed_libraries[swap_pos]
 
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
 
        # Process libraries before swap point
        for i in range(swap_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Safety check
                continue
            library = data.libs[lib_id]
            curr_time += library.signup_days
 
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
 
        # Re-process from swap point
        i = swap_pos
        while i < len(new_solution.signed_libraries):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Skip invalid library IDs
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
                continue
 
            library = data.libs[lib_id]
 
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.extend(new_solution.signed_libraries[i:])
                new_solution.signed_libraries = new_solution.signed_libraries[:i]
                break
 
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
 
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
 
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
                i += 1
            else:
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
 
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
 
        return new_solution
    
    def tweak_solution_insert_library(self, solution, data, target_lib=None):
        if not solution.unsigned_libraries and target_lib is None:
            return solution
 
        new_solution = copy.deepcopy(solution)
        curr_time = sum(data.libs[lib_id].signup_days for lib_id in new_solution.signed_libraries)
 
        if target_lib is not None and target_lib not in new_solution.signed_libraries:
            lib_to_insert = target_lib
        else:
            if not new_solution.unsigned_libraries:
                return solution
            insert_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)
            lib_to_insert = new_solution.unsigned_libraries[insert_idx]
            new_solution.unsigned_libraries.pop(insert_idx)
 
        if curr_time + data.libs[lib_to_insert].signup_days >= data.num_days:
            return solution
 
        time_left = data.num_days - (curr_time + data.libs[lib_to_insert].signup_days)
        max_books_scanned = time_left * data.libs[lib_to_insert].books_per_day
 
        available_books = sorted(
            {book.id for book in data.libs[lib_to_insert].books} - new_solution.scanned_books,
            key=lambda b: -data.scores[b]
        )[:max_books_scanned]
 
        if available_books:
            best_pos = len(new_solution.signed_libraries)
            best_score = 0
            best_solution = None
 
            for pos in range(len(new_solution.signed_libraries) + 1):
                test_solution = copy.deepcopy(new_solution)
                test_solution.signed_libraries.insert(pos, lib_to_insert)
                test_solution.scanned_books_per_library[lib_to_insert] = available_books
                test_solution.scanned_books.update(available_books)
                test_solution.calculate_fitness_score(data.scores)
 
                if test_solution.fitness_score > best_score:
                    best_score = test_solution.fitness_score
                    best_pos = pos
                    best_solution = test_solution
 
            return best_solution if best_solution else solution
 
        return solution
    
    def tweak_solution_swap_last_book_tabu(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution, None

        chosen_lib_id = random.choice(list(solution.scanned_books_per_library.keys()))
        scanned_books = solution.scanned_books_per_library.get(chosen_lib_id, [])

        if not scanned_books:
            return solution, None

        last_scanned_book = scanned_books[-1]
        library_dict = {lib.id: lib for lib in data.libs}

        best_book = None
        best_score = -1
        best_unsigned_lib = None

        for unsigned_lib_id in solution.unsigned_libraries:
            library = library_dict[unsigned_lib_id]
            for book in library.books:
                if book.id not in solution.scanned_books:
                    if data.scores[book.id] > best_score:
                        best_book = book.id
                        best_score = data.scores[book.id]
                        best_unsigned_lib = unsigned_lib_id
                    break

        if best_book is None:
            return solution, None

        # Clone solution (faster than deepcopy if you have a .clone() method)
        new_solution = solution.clone()

        new_solution.scanned_books_per_library[chosen_lib_id].remove(last_scanned_book)
        new_solution.scanned_books_per_library[chosen_lib_id].append(best_book)
        new_solution.scanned_books.remove(last_scanned_book)
        new_solution.scanned_books.add(best_book)

        new_solution.calculate_fitness_score(data.scores)

        move_signature = ('swap_last_book', chosen_lib_id, best_unsigned_lib, last_scanned_book, best_book)

        return new_solution, move_signature

    def tweak_solution_swap_signed_tabu(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution, None

        book_count = defaultdict(int)
        unscanned_books_per_library = {}

        # Collect unscanned books per library and book count
        for library in data.libs:
            if library.id in solution.signed_libraries:
                unsigned_books = []
                for book in library.books:
                    book_count[book.id] += 1
                    if book.id not in solution.scanned_books_per_library.get(library.id, []) and book.id not in solution.scanned_books:
                        unsigned_books.append(book.id)
                if unsigned_books:
                    unscanned_books_per_library[library.id] = unsigned_books

        if len(unscanned_books_per_library) == 1:
            return solution, None  # No need to swap if only one library has unscanned books

        possible_books = [
            book_id for book_id, count in book_count.items()
            if count > 1 and book_id in solution.scanned_books
        ]

        # Filter valid books for swap
        valid_books = set()
        for library, unscanned_books in unscanned_books_per_library.items():
            for book_id in possible_books:
                for book in data.libs[library].books:
                    if book.id == book_id:
                        valid_books.add(book_id)

        if not valid_books:
            return solution, None  # No valid books to swap

        # Randomly select a book to swap
        book_to_move = random.choice(list(valid_books))

        # Find current library of the book
        current_library = None
        for lib_id, books in solution.scanned_books_per_library.items():
            if book_to_move in books:
                current_library = lib_id
                break

        if current_library is None or unscanned_books_per_library.get(current_library) is None or not unscanned_books_per_library[current_library]:
            return solution, None  # No valid library to move the book from

        # Find a library to move the book to
        possible_libraries = [
            lib for lib in data.book_libs[book_to_move]
            if lib != current_library and any(
                library.id == lib and any(book.id not in solution.scanned_books for book in library.books)
                for library in data.libs if library.id in solution.signed_libraries
            )
        ]

        if not possible_libraries:
            return solution, None  # No valid libraries to move the book to

        # Select a new library for the book
        new_library = random.choice(possible_libraries)

        # Clone solution (faster than deepcopy if you have a .clone() method)
        new_solution = solution.clone()

        # Remove the book from the current library
        new_solution.scanned_books_per_library[current_library].remove(book_to_move)
        new_solution.scanned_books.remove(book_to_move)

        # Add the book to the new library, maintaining feasibility
        current_books_in_new_library = new_solution.scanned_books_per_library[new_library]
        max_books_per_day = data.libs[new_library].books_per_day
        days_before_sign_up = sum(data.libs[id].signup_days for id in new_solution.signed_libraries if id == new_library)
        num_of_days_available = data.num_days - days_before_sign_up

        # Ensure feasibility: Check if new library has space
        if len(current_books_in_new_library) > num_of_days_available * max_books_per_day:
            book_to_remove = random.choice(list(current_books_in_new_library))
            current_books_in_new_library.remove(book_to_remove)
            new_solution.scanned_books.remove(book_to_remove)

        # Add the book to the new library
        current_books_in_new_library.append(book_to_move)
        new_solution.scanned_books.add(book_to_move)

        # Find an unscanned book in the current library to replace
        books_in_current_library = new_solution.scanned_books_per_library[current_library]
        new_scanned_book = random.choice(unscanned_books_per_library.get(current_library))
        books_in_current_library.append(new_scanned_book)
        new_solution.scanned_books.add(new_scanned_book)

        # Calculate delta fitness for the swap
        new_solution.calculate_fitness_score(data.scores)

        # Move signature (describes the swap action)
        move_signature = ('swap_signed_book', current_library, new_library, book_to_move, new_scanned_book)

        return new_solution, move_signature

    def tweak_solution_swap_signed_with_unsigned_tabu(self, solution, data, bias_type=None, bias_ratio=2 / 3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution, None

        local_signed_libs = solution.signed_libraries.copy()
        local_unsigned_libs = solution.unsigned_libraries.copy()

        total_signed = len(local_signed_libs)

        # Bias
        if bias_type == "favor_first_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(local_unsigned_libs) - 1)

        signed_lib_id = local_signed_libs[signed_idx]
        unsigned_lib_id = local_unsigned_libs[unsigned_idx]

        # Swap the libraries
        local_signed_libs[signed_idx] = unsigned_lib_id
        local_unsigned_libs[unsigned_idx] = signed_lib_id

        # Preserve the part before `signed_idx`
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        lib_lookup = {lib.id: lib for lib in data.libs}

        # Process libraries before the swapped index
        for i in range(signed_idx):
            lib_id = solution.signed_libraries[i]
            library = lib_lookup.get(lib_id)

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Recalculate from `signed_idx` onward
        new_signed_libraries = local_signed_libs[:signed_idx]

        for i in range(signed_idx, len(local_signed_libs)):
            lib_id = local_signed_libs[i]
            library = lib_lookup.get(lib_id)

            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(library.id)
                continue

            curr_time += library.signup_days
            time_left = data.num_days - curr_time
            max_books_scanned = time_left * library.books_per_day

            available_books = [book.id for book in library.books if book.id not in scanned_books][:max_books_scanned]

            if available_books:
                new_signed_libraries.append(library.id)
                new_scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)

        # Update solution
        new_solution = Solution(new_signed_libraries, local_unsigned_libs, new_scanned_books_per_library, scanned_books)
        new_solution.calculate_fitness_score(data.scores)

        # Move signature (describes the swap action)
        move_signature = ('swap_signed_with_unsigned', signed_lib_id, unsigned_lib_id)

        return new_solution, move_signature

    def tweak_solution_swap_same_books_tabu(self, solution, data):
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]

        if len(library_ids) < 2:
            return solution, None

        idx1 = random.randint(0, len(library_ids) - 1)
        idx2 = random.randint(0, len(library_ids) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(library_ids) - 1)

        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]

        ordered_libs = [data.libs[lib_id] for lib_id in library_ids]

        all_lib_ids = set(range(len(data.libs)))
        remaining_lib_ids = all_lib_ids - set(library_ids)
        for lib_id in sorted(remaining_lib_ids):
            ordered_libs.append(data.libs[lib_id])

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in ordered_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        # Create the new solution
        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        new_solution.calculate_fitness_score(data.scores)

        # Move signature (describes the swap action)
        move_signature = ('swap_same_books', library_ids[idx1], library_ids[idx2])

        return new_solution, move_signature

    def tweak_solution_insert_library_tabu(self, solution, data, target_lib=None):
        if not solution.unsigned_libraries and target_lib is None:
            return solution, None

        new_solution = copy.deepcopy(solution)
        curr_time = sum(data.libs[lib_id].signup_days for lib_id in new_solution.signed_libraries)

        if target_lib is not None and target_lib not in new_solution.signed_libraries:
            lib_to_insert = target_lib
        else:
            if not new_solution.unsigned_libraries:
                return solution, None
            insert_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)
            lib_to_insert = new_solution.unsigned_libraries.pop(insert_idx)

        if curr_time + data.libs[lib_to_insert].signup_days >= data.num_days:
            return solution, None

        time_left = data.num_days - (curr_time + data.libs[lib_to_insert].signup_days)
        max_books_scanned = time_left * data.libs[lib_to_insert].books_per_day

        available_books = sorted(
            {book.id for book in data.libs[lib_to_insert].books} - new_solution.scanned_books,
            key=lambda b: -data.scores[b]
        )[:max_books_scanned]

        if available_books:
            best_pos = len(new_solution.signed_libraries)
            best_score = 0
            best_solution = None

            for pos in range(len(new_solution.signed_libraries) + 1):
                test_solution = copy.deepcopy(new_solution)
                test_solution.signed_libraries.insert(pos, lib_to_insert)
                test_solution.scanned_books_per_library[lib_to_insert] = available_books
                test_solution.scanned_books.update(available_books)
                test_solution.calculate_fitness_score(data.scores)

                if test_solution.fitness_score > best_score:
                    best_score = test_solution.fitness_score
                    best_pos = pos
                    best_solution = test_solution

            if best_solution:
                move_signature = ("insert_library", lib_to_insert, best_pos)
                return best_solution, move_signature

        return solution, None

    def tweak_solution_swap_neighbor_libraries_tabu(self, solution, data):
        """Swaps two adjacent libraries in the signed list to create a neighbor solution."""
        if len(solution.signed_libraries) < 2:
            return solution, None

        new_solution = copy.deepcopy(solution)
        swap_pos = random.randint(0, len(new_solution.signed_libraries) - 2)

        # Swap adjacent libraries
        lib1 = new_solution.signed_libraries[swap_pos]
        lib2 = new_solution.signed_libraries[swap_pos + 1]
        new_solution.signed_libraries[swap_pos], new_solution.signed_libraries[swap_pos + 1] = lib2, lib1

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        # Process libraries before swap point
        for i in range(swap_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Safety check
                continue
            library = data.libs[lib_id]
            curr_time += library.signup_days

            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)

        # Re-process from swap point
        i = swap_pos
        while i < len(new_solution.signed_libraries):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):  # Skip invalid library IDs
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)
                continue

            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.extend(new_solution.signed_libraries[i:])
                new_solution.signed_libraries = new_solution.signed_libraries[:i]
                break

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
                i += 1
            else:
                new_solution.unsigned_libraries.append(lib_id)
                new_solution.signed_libraries.pop(i)

        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)

        move_signature = ('swap_neighbor', lib1, lib2)
        return new_solution, move_signature
