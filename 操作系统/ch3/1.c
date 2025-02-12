#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/wait.h>

#define MAX_LINE 80 /* 80 chars per line, per command */

int main(void) {
    char *args[MAX_LINE / 2 + 1]; /* command line (of 80) has max of 40 arguments */
    char input[MAX_LINE];         /* stores the command input */
    int should_run = 1;           /* flag to determine when to exit program */
    int background;               /* flag for background process */

    while (should_run) {
        printf("osh>");
        fflush(stdout);

        /* Step 1: Read user input */
        fgets(input, MAX_LINE, stdin); /* read command input */
        input[strlen(input) - 1] = '\0'; /* remove newline character */

        /* Step 2: Parse the input into args array */
        char *token = strtok(input, " ");
        int i = 0;
        background = 0;

        while (token != NULL) {
            args[i] = token;
            i++;
            token = strtok(NULL, " ");
        }
        args[i] = NULL; /* null terminate the args array */

        /* Check if the last argument is & */
        if (i > 0 && strcmp(args[i - 1], "&") == 0) {
            background = 1;  /* Set background flag */
            args[i - 1] = NULL; /* Remove & from args */
        }

        /* Step 3: Fork a child process */
        pid_t pid = fork();
        if (pid < 0) { /* error occurred */
            fprintf(stderr, "Fork failed\n");
            return 1;
        } else if (pid == 0) { /* child process */
            /* Step 4: Execute the command using execvp */
            if (execvp(args[0], args) == -1) {
                perror("osh");
            }
            exit(1);
        } else { /* parent process */
            if (!background) {
                /* Step 5: Parent waits for child to complete if not background */
                wait(NULL);
            }
        }
    }

    return 0;
}
