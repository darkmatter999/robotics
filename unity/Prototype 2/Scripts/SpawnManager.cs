using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnManager : MonoBehaviour
{
    // Create an array of different animals
    public GameObject[] animalPrefabs;
    // Start is called before the first frame update
    void Start()
    {
        // Invoke the SpawnRandomAnimal after [startDelay] has passed for an interval of [spawnInterval] seconds
        InvokeRepeating("SpawnRandomAnimal", startDelay, spawnInterval);
    }

    private float spawnRangeX = 15;
    private float spawnPosZ = 20;
    //Spawning action starts after 2 seconds
    private float startDelay = 2;
    // Random animals are spawned at an interval of 1.5 seconds
    private float spawnInterval = 1.5f;

    // Update is called once per frame
    void Update()
    {
        
    }

    void SpawnRandomAnimal()
    {
        // Spawn random animals at random x positions
        Vector3 spawnPos = new Vector3(Random.Range(-spawnRangeX, spawnRangeX), 0, spawnPosZ);
        int animalIndex = Random.Range(0, animalPrefabs.Length);

        Instantiate(animalPrefabs[animalIndex], spawnPos, animalPrefabs[animalIndex].transform.rotation);
    }
}
