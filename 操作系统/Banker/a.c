#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define NUMBER_OF_CUSTOMERS 5
#define NUMBER_OF_RESOURCES 4

int available[NUMBER_OF_RESOURCES];
int maximum[NUMBER_OF_CUSTOMERS][NUMBER_OF_RESOURCES];
int allocation[NUMBER_OF_CUSTOMERS][NUMBER_OF_RESOURCES];
int need[NUMBER_OF_CUSTOMERS][NUMBER_OF_RESOURCES];

void initialize_arrays(int argc, char *argv[], const char *filename);
int request_resources(int customer_num, int request[]);
int release_resources(int customer_num, int release[]);
int is_safe_state();
void print_status();

int main(int argc, char *argv[])
{
    if (argc < NUMBER_OF_RESOURCES + 1)
    {
        printf("Usage: %s <resource1> <resource2> ... <resourceN>\n", argv[0]);
        exit(1);
    }

    // Initialize available array and other arrays from file
    initialize_arrays(argc, argv, "maximum.txt");

    char command;
    int customer_num, resources[NUMBER_OF_RESOURCES];

    while (true)
    {
        printf("Enter command (RQ <cust_num> <r1> <r2> ... or RL <cust_num> <r1> <r2> ... or * to display status): ");
        scanf(" %c", &command);

        if (command == '*')
        {
            print_status();
        }
        else if (command == 'R')
        {
            scanf("%c", &command);
            if (command == 'Q')
            {
                scanf(" %d", &customer_num);
                for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
                {
                    scanf("%d", &resources[i]);
                }
                if (request_resources(customer_num, resources) == 0)
                {
                    printf("Request granted.\n");
                }
                else
                {
                    printf("Request denied.\n");
                }
            }
            if (command == 'L')
            {
                scanf("L %d", &customer_num);
                for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
                {
                    scanf("%d", &resources[i]);
                }
                release_resources(customer_num, resources);
                printf("Resources released.\n");
            }
        }
        else
        {
            printf("Invalid command.\n");
        }
    }

    return 0;
}

void initialize_arrays(int argc, char *argv[], const char *filename)
{
    // Initialize available resources from command line
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        available[i] = atoi(argv[i + 1]);
    }

    // Read maximum matrix from file
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(1);
    }

    for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        for (int j = 0; j < NUMBER_OF_RESOURCES; j++)
        {
            if (j == NUMBER_OF_RESOURCES - 1)
            {
                // 最后一列不需要逗号
                fscanf(file, "%d", &maximum[i][j]);
            }
            else
            {
                // 处理逗号分隔符
                fscanf(file, "%d,", &maximum[i][j]);
            }
            allocation[i][j] = 0;
            need[i][j] = maximum[i][j];
        }
    }

    fclose(file);
}

int request_resources(int customer_num, int request[])
{
    // Check if request is within need
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        if (request[i] > need[customer_num][i])
        {
            return -1; // Request exceeds need
        }
    }

    // Check if resources are available
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        if (request[i] > available[i])
        {
            return -1; // Not enough resources
        }
    }

    // Pretend to allocate resources
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        available[i] -= request[i];
        allocation[customer_num][i] += request[i];
        need[customer_num][i] -= request[i];
    }

    // Check if the system is in a safe state
    if (is_safe_state())
    {
        return 0; // Request granted
    }
    else
    {
        // Rollback allocation if state is unsafe
        for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
        {
            available[i] += request[i];
            allocation[customer_num][i] -= request[i];
            need[customer_num][i] += request[i];
        }
        return -1; // Request denied
    }
}

int release_resources(int customer_num, int release[])
{
    // Release resources
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        allocation[customer_num][i] -= release[i];
        available[i] += release[i];
        need[customer_num][i] += release[i];
    }
    return 0;
}

int is_safe_state()
{
    int work[NUMBER_OF_RESOURCES];
    bool finish[NUMBER_OF_CUSTOMERS] = {false};

    // Initialize work = available
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        work[i] = available[i];
    }

    while (true)
    {
        bool found = false;
        for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
        {
            if (!finish[i])
            {
                int j;
                for (j = 0; j < NUMBER_OF_RESOURCES; j++)
                {
                    if (need[i][j] > work[j])
                        break;
                }
                if (j == NUMBER_OF_RESOURCES)
                {
                    for (int k = 0; k < NUMBER_OF_RESOURCES; k++)
                    {
                        work[k] += allocation[i][k];
                    }
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found)
            break;
    }

    for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        if (!finish[i])
        {
            printf("Not Safe.\n");
            return 0; // Not safe
        }
    }
    printf("Safe.\n");
    return 1; // Safe
}

void print_status()
{
    printf("Available resources:\n");
    for (int i = 0; i < NUMBER_OF_RESOURCES; i++)
    {
        printf("%d ", available[i]);
    }
    printf("\n\nMaximum resources:\n");
    for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        for (int j = 0; j < NUMBER_OF_RESOURCES; j++)
        {
            printf("%d ", maximum[i][j]);
        }
        printf("\n");
    }
    printf("\nAllocation:\n");
    for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        for (int j = 0; j < NUMBER_OF_RESOURCES; j++)
        {
            printf("%d ", allocation[i][j]);
        }
        printf("\n");
    }
    printf("\nNeed:\n");
    for (int i = 0; i < NUMBER_OF_CUSTOMERS; i++)
    {
        for (int j = 0; j < NUMBER_OF_RESOURCES; j++)
        {
            printf("%d ", need[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
