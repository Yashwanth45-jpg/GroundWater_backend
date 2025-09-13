//api ke andar kya hoga aur kese hoga usske kam athi hai

const userModel = require('../models/User.models')
const jwt = require('jsonwebtoken')
const bcrypt = require('bcrypt')

async function registerController(req, res) {
const {username, password} = req.body;

    const ifuserExits = await userModel.findOne({
        username
    })

    if(ifuserExits) {
        return res.status(409).json({
            msg:"user Already Exits",
        })
    }

    const  user = await userModel.create({
        username, 
        password: await bcrypt.hash(password, 10)
    })
    
    const token = jwt.sign({
        id:user._id
    },process.env.JWT_SECRET_KEY)

    res.cookie('token', token, { 
        httpOnly: true,
        secure: false, // Set to true in production with HTTPS
        sameSite: 'lax'
    })

    res.status(201).json({
        msg:"user created successfully",
        user
    })
}


async function loginController(req, res) {
    const {username, password} = req.body;

    const user = await userModel.findOne({
        username
    })

    if(!user) {
        return res.status(400).json({
            msg:"user not found"
        })
    }

    const isPasswordExists = await bcrypt.compare(password, user.password);

    if(!isPasswordExists) {
        return res.status(400).json({
            msg:"Password Incorrect"
        })
    }

    const token = jwt.sign({id:user._id}, process.env.JWT_SECRET_KEY)

    res.cookie("token", token, { 
        httpOnly: true,
        secure: false, // Set to true in production with HTTPS
        sameSite: 'lax'
    })

    res.status(200).json({
        msg:"user login successfully",
        user
    })
}

async function logoutController(req, res) {
    res.clearCookie('token', {
        httpOnly: true,
        secure: false,
        sameSite: 'lax'
    })
    
    res.status(200).json({
        msg: "Logged out successfully"
    })
}

module.exports = {registerController, loginController, logoutController}