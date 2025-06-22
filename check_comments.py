import sys
import tokenize

def check_comments(filename):
    """
    Checks a Python file for non-ASCII characters in comments.
    """
    has_non_ascii_comment = False
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            tokens = tokenize.generate_tokens(f.readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_text = token.string
                    if not comment_text.isascii():
                        print(
                            f"Error: Non-English (non-ASCII) comment found in "
                            f"{filename} on line {token.start[0]}:"
                        )
                        print(f"  '{comment_text}'")
                        has_non_ascii_comment = True
        except tokenize.TokenError:
            # This can happen if the file is not valid Python.
            # We can ignore these files.
            pass
    return has_non_ascii_comment

if __name__ == "__main__":
    exit_code = 0
    for filename in sys.argv[1:]:
        if check_comments(filename):
            exit_code = 1
    sys.exit(exit_code) 