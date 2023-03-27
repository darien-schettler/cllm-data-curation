<br>

# Downloading and Curating The Stack Dataset

---

### To Do

- [x] Add a script to download the dataset
- [x] Test **git-lfs** method for script download
- [x] Test **requests** for script download
- [x] Test **HuggingFace 🤗** for script download

---

### **Other**

#### <u>Dataset Structure</u>

##### **DATA INSTANCES**
Each data instance corresponds to one file. The content of the file is in the `content` feature, and other features (`repository_name`, `licenses`, etc.) provide some metadata. Note that a given file can appear in several different repositories that satisfy our safe-license criterion. If that is the case, only the first – in alphabetical order -- of these repositories is shown for simplicity.

##### **DATA FIELDS**
- `content` (string): the content of the file.
- `repository_name` (string): name of the repository. If a file appears in several repositories that satisfy our license criterion, it will only show the first in alphabetical order.
- `licenses` (list of strings): list of licenses that were detected in the repository. They will all be "safe licenses".
- `path` (string): relative path in the repository.
- `size` (integer): size of the uncompressed file.
- `lang` (string): the programming language. 
- `ext` (string): file extension
- `avg_line_length` (float): the average line-length of the file.
- `max_line_length` (integer): the maximum line-length of the file.
- `alphanum_fraction` (float): the fraction of characters in the file that are alphabetical or numerical characters.
- `hexsha` (string): unique git hash of file
- `max_{stars|forks|issues}_repo_path` (string): path to file in repo containing this file with maximum number of `{stars|forks|issues}`
- `max_{stars|forks|issues}_repo_name` (string): name of repo containing this file with maximum number of `{stars|forks|issues}`
- `max_{stars|forks|issues}_repo_head_hexsha` (string): hexsha of repository head
- `max_{stars|forks|issues}_repo_licenses` (string): licenses in repository 
- `max_{stars|forks|issues}_count` (integer): number of `{stars|forks|issues}` in repository
- `max_{stars|forks|issues}_repo_{stars|forks|issues}_min_datetime` (string): first timestamp of a `{stars|forks|issues}` event
- `max_{stars|forks|issues}_repo_{stars|forks|issues}_max_datetime` (string): last timestamp of a `{stars|forks|issues}` event