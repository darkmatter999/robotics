using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 15.0f;
    private float randomYStartPos;
    public float sensorOffset = 0.2f;
    public float upperBorderPos = 2.8f;
    public float lowerBorderPos = -4.75f;
    public Vector3 upperBorderOffset = new Vector3(0, -0.2f, 0);
    public Vector3 lowerBorderOffset = new Vector3(0, 0.2f, 0);
    public GameObject sensorUpper;
    public GameObject sensorMiddle;
    public GameObject sensorLower;
    // Start is called before the first frame update
    void Start()
    {
        randomYStartPos = Random.Range(-4.5f, 2.5f);
        transform.position = new Vector3(transform.position.x, randomYStartPos, transform.position.z);
    }

    // Update is called once per frame
    void Update()
    {
        sensorUpper.transform.position = new Vector3(sensorUpper.transform.position.x, transform.position.y + sensorOffset, sensorUpper.transform.position.z);
        sensorMiddle.transform.position = new Vector3(sensorMiddle.transform.position.x, transform.position.y, sensorMiddle.transform.position.z);
        sensorLower.transform.position = new Vector3(sensorLower.transform.position.x, transform.position.y - sensorOffset, sensorLower.transform.position.z);
        
        // Both mouse or keyboard movement are feasible
        // For keyboard movement use if (Input.GetKey(KeyCode.UpArrow/DownArrow))
        if (Input.GetAxis("Mouse Y")>0)
        {
            if (transform.position.y < upperBorderPos)
            {
                transform.Translate(Vector3.up * moveSpeed * Time.deltaTime);
            }
            else
            {
                // Movement stops after border is reached
                transform.Translate(Vector3.up * 0);
            }
        }

        if (Input.GetAxis("Mouse Y")<0)
        {
            if (transform.position.y > lowerBorderPos)
            {
                transform.Translate(Vector3.down * moveSpeed * Time.deltaTime);
            }
            else
            {
                transform.Translate(Vector3.down * 0);
            }
        }
    }
}

// See comments in EnemyController
