#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_PHILOSOPHERS 5

enum { THINKING, HUNGRY, EATING } state[NUM_PHILOSOPHERS];
pthread_mutex_t mutex;
pthread_cond_t condition[NUM_PHILOSOPHERS];
void pickup_forks(int philosopher_number) {
    pthread_mutex_lock(&mutex);
    state[philosopher_number] = HUNGRY;
    test(philosopher_number);
    while (state[philosopher_number] != EATING) {
        pthread_cond_wait(&condition[philosopher_number], &mutex);
    }
    pthread_mutex_unlock(&mutex);
}
void return_forks(int philosopher_number) {
    pthread_mutex_lock(&mutex);
    state[philosopher_number] = THINKING;
    test((philosopher_number + 4) % NUM_PHILOSOPHERS); // Left neighbor
    test((philosopher_number + 1) % NUM_PHILOSOPHERS); // Right neighbor
    pthread_mutex_unlock(&mutex);
}
void test(int philosopher_number) {
    if (state[philosopher_number] == HUNGRY &&
        state[(philosopher_number + 4) % NUM_PHILOSOPHERS] != EATING &&
        state[(philosopher_number + 1) % NUM_PHILOSOPHERS] != EATING) {
        state[philosopher_number] = EATING;
        pthread_cond_signal(&condition[philosopher_number]);
    }
}
void* philosopher(void* num) {
    int id = *(int*)num;
    while (1) {
        // Think
        printf("Philosopher %d is thinking.\n", id);
        sleep(rand() % 3 + 1);

        // Try to pick up forks and eat
        pickup_forks(id);
        printf("Philosopher %d is eating.\n", id);
        sleep(rand() % 3 + 1);
        
        // Return forks and resume thinking
        return_forks(id);
    }
}
int main() {
    pthread_t philosophers[NUM_PHILOSOPHERS];
    pthread_mutex_init(&mutex, NULL);
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_cond_init(&condition[i], NULL);
    }

    int ids[NUM_PHILOSOPHERS];
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        ids[i] = i;
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }

    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_join(philosophers[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_cond_destroy(&condition[i]);
    }
    return 0;
}
