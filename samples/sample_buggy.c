#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void handle_input(char *user_input) {
    char buffer[32];
    char *heap_buf = (char *)malloc(128);
    gets(buffer);
    strcpy(buffer, user_input);
    printf("%s\n", heap_buf);
}
