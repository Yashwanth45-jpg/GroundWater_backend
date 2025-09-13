const express = require('express');
const {registerController, loginController, logoutController} = require('../controllers/Auth.controller')

const router  = express.Router()

//routes kon kon se hai vo iskeliye kam athi hai


/*
post -> /posts api
post ->/register
post -> /user [protected]
post -> /logout

*/

router.post('/register', registerController)

router.post('/login', loginController)

router.post('/logout', logoutController)

module.exports = router