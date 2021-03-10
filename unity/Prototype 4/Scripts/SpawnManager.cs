using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnManager : MonoBehaviour
{
    public GameObject enemyPrefab;
    private float spawnRange = 9.0f;
    public int enemyCount;
    public int initEnemies = 2;
    public GameObject powerup;
    // Start is called before the first frame update
    void Start()
    {
        SpawnEnemyWave(initEnemies);
        Instantiate(powerup, GenerateSpawnPosition(), powerup.transform.rotation);
    }

    // Update is called once per frame
    void Update()
    {
        // Each frame, we count how many enemies are still there. If none, we spawn a new one
        enemyCount = FindObjectsOfType<Enemy>().Length;
        if (enemyCount == 0)
        {
            initEnemies++;
            SpawnEnemyWave(initEnemies);
            Instantiate(powerup, GenerateSpawnPosition(), powerup.transform.rotation);
        }
    }

    void SpawnEnemyWave(int enemiesToSpawn)
    {
        for (int i = 0; i < enemiesToSpawn; i++)
        {
            Instantiate(enemyPrefab, GenerateSpawnPosition(), enemyPrefab.transform.rotation);
        }
    }

    private Vector3 GenerateSpawnPosition()
    {
        float spawnPosX = Random.Range(-spawnRange, spawnRange);
        float spawnPosZ = Random.Range(-spawnRange, spawnRange);
        Vector3 spawnPos = new Vector3(spawnPosX, 0, spawnPosZ);
        return spawnPos;
    }
}
