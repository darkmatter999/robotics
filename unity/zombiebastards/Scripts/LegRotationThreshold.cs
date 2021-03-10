using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LegRotationThreshold : MonoBehaviour
{
    
    private float rotationThreshold = -0.2f;
    private float currentRotation;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        currentRotation = transform.rotation.x;
        if (currentRotation < rotationThreshold)
        {
            transform.Rotate(0, 0, 0);
        }
    }
}
