using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class calcCaller : MonoBehaviour
{
    private calc calcExec;
    // Start is called before the first frame update
    void Start()
    {
        calcExec = GetComponent<calc>();
        calcExec.Calc();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

// This script is for testing function calling from another class. Here, we call the public method 'Calc', which calculates 1+1, from the
// public class 'calc'. 'calcCaller' first initializes a variable 'calcExec' based on the class 'calc', then gets the class component and
// finally calls the method 'Calc'.
