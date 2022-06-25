#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int tempValue = 0;
int MaxValue = 0;

int main(void){
    srand((unsigned)time(NULL));
    for(int i=0;i<1000;i++){
        if(tempValue == 0){
            tempValue = rand();
            MaxValue = tempValue;
            printf("Spawning Begin Value with : %d",tempValue);
        } else {
            if(tempValue > MaxValue){              
                printf("Replacing Current MaxValue : %d with new Value : %d.\n",MaxValue,tempValue);
                MaxValue = tempValue;
            } else if (tempValue == MaxValue){
                printf("Current Value is tied with MaxValue,skip!\n");
            } else if (tempValue < MaxValue){
                printf("Current Value : %d is less than MaxValue : %d,ignored!\n",tempValue,MaxValue);
            }
            tempValue = rand();
        }
    }
    printf("Got Final Maximum Value : %d\n",MaxValue);
}